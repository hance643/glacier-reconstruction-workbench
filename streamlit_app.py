from __future__ import annotations

from datetime import datetime
from io import BytesIO
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

import numpy as np
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import trimesh

from glacier.pipeline import PipelineConfig, run_colmap_openmvs, run_vggt, validate_tools
from glacier.preprocessing import PreprocessConfig, preprocess, read_rgb, write_png


st.set_page_config(page_title="冰川三维重建工作台", page_icon="🏔️", layout="wide")

wallpaper_path = Path(__file__).resolve().parent / "assets" / "snow-peaks-wallpaper.jpg"
wallpaper_css = ""
if wallpaper_path.is_file():
    import base64
    wallpaper_data = base64.b64encode(wallpaper_path.read_bytes()).decode("ascii")
    wallpaper_css = f"background-image: linear-gradient(110deg, rgba(5, 10, 18, .91), rgba(7, 15, 26, .77)), url(data:image/jpeg;base64,{wallpaper_data});"

st.markdown(f"""
<style>
:root {{ color-scheme: dark; }}
.stApp {{ {wallpaper_css} background-color:#05080e; background-size:cover; background-attachment:fixed; background-position:center; color:#f1f6fc; }}
[data-testid="stHeader"] {{ background:rgba(5,8,14,.62); }}
.block-container {{ max-width:1480px; padding-top:2.1rem; padding-bottom:4rem; }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,rgba(7,13,23,.96),rgba(10,18,29,.88)); border-right:1px solid rgba(218,235,250,.12); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#aebdcb; }}
h1,h2,h3 {{ color:#f5f9fd !important; letter-spacing:-.025em; }}
h1 {{ font-size:2.35rem !important; font-weight:750 !important; }}
h2 {{ font-size:1.45rem !important; margin-top:1.8rem !important; }}
[data-testid="stCaptionContainer"] {{ color:#9aabba; }}
[data-testid="stAlert"] {{ background:rgba(18,29,43,.76); border:1px solid rgba(174,199,217,.24); border-radius:14px; color:#dce8f2; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:rgba(12,20,31,.67); border:1px solid rgba(220,237,250,.14); border-radius:18px; backdrop-filter:blur(14px); }}
[data-testid="stFileUploader"] section {{ background:rgba(13,20,32,.62); border:1px dashed rgba(174,199,217,.45); border-radius:16px; }}
[data-testid="stFileUploader"] section:hover {{ border-color:#aec7d9; }}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{ min-height:2.7rem; color:#f4f8fc; background:linear-gradient(135deg,rgba(39,60,82,.9),rgba(20,32,48,.94)); border:1px solid rgba(174,199,217,.44); border-radius:999px; padding:0 1.4rem; transition:all .18s ease; }}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {{ border-color:#d3e4f1; background:rgba(54,76,99,.95); transform:translateY(-1px); }}
[data-testid="stBaseButton-primary"] {{ background:linear-gradient(135deg,#c8ddea,#8eafc5) !important; color:#101a24 !important; border:0 !important; font-weight:750 !important; }}
[data-testid="stBaseButton-primary"] p {{ color:#101a24 !important; }}
input, textarea, [data-baseweb="select"] > div {{ background-color:rgba(13,20,32,.88) !important; border-color:rgba(130,151,171,.4) !important; border-radius:10px !important; color:#f1f6fc !important; }}
[data-testid="stExpander"] {{ background:rgba(12,20,31,.56); border:1px solid rgba(220,237,250,.13); border-radius:14px; }}
[data-testid="stMetric"] {{ background:rgba(17,27,40,.64); border:1px solid rgba(220,237,250,.13); padding:14px 16px; border-radius:14px; }}
[data-testid="stPlotlyChart"] {{ background:rgba(10,16,25,.75); border:1px solid rgba(220,237,250,.13); border-radius:18px; padding:8px; }}
hr {{ border-color:rgba(220,237,250,.13); }}
@media (max-width: 760px) {{ .block-container {{ padding:1.2rem 1rem 3rem; }} h1 {{ font-size:1.8rem !important; }} }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🏔️ 冰川重建")
    st.caption("GLACIER RECONSTRUCTION STUDIO")
    st.markdown("---")
    st.markdown("**工作流程**")
    st.markdown("影像导入　·　预处理　·　重建　·　模型预览")
    st.markdown("---")
    st.caption("网页演示版")

st.title("冰川三维重建工作台")
st.caption("网页演示版 · 照片预处理、可用重建路径检测与三维模型预览")
st.info(
    "这是独立的 Streamlit 网页界面。它可以在当前服务器提供的环境中运行重建；"
    "Streamlit Cloud 默认没有 VGGT、COLMAP、OpenMVS 和 GPU，需另行部署这些工具后才能运行完整重建。"
)


def pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        vggt_python=os.getenv("VGGT_PYTHON", ""),
        vggt_repo=os.getenv("VGGT_REPO", ""),
        vggt_args=os.getenv("VGGT_ARGS", "[]"),
        colmap=os.getenv("COLMAP_BIN", "colmap"),
        openmvs_bin=os.getenv("OPENMVS_BIN", ""),
        colmap_extras=os.getenv("COLMAP_EXTRAS", "{}"),
        openmvs_extras=os.getenv("OPENMVS_EXTRAS", "{}"),
    )


def available_methods():
    config = pipeline_config()
    available = {}
    for method, label in (("A", "VGGT"), ("B", "COLMAP + OpenMVS")):
        try:
            validate_tools([method], config)
            available[method] = label
        except Exception as exc:
            available[method] = str(exc)
    return config, available


config, method_status = available_methods()
with st.expander("重建环境状态", expanded=False):
    for method, label in (("A", "VGGT"), ("B", "COLMAP + OpenMVS")):
        reason = method_status[method]
        if reason in {"VGGT", "COLMAP + OpenMVS"}:
            st.success(f"路径 {method} · {reason}：已检测到所需程序")
        else:
            st.warning(f"路径 {method}：当前不可运行。{reason}")
    st.caption("可在托管服务器的环境变量中配置 VGGT_PYTHON、VGGT_REPO、COLMAP_BIN 和 OPENMVS_BIN。")

st.header("1. 上传照片")
uploads = st.file_uploader(
    "选择同一场景的多张照片（至少两张），支持 JPG、PNG、TIFF、BMP、WebP",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
    accept_multiple_files=True,
    key="photo_uploads",
)
if uploads:
    st.write(f"已选择 {len(uploads)} 张照片")
    preview_cols = st.columns(min(4, len(uploads)))
    for index, upload in enumerate(uploads[:4]):
        try:
            preview_cols[index % len(preview_cols)].image(
                upload.getvalue(), caption=upload.name, use_container_width=True
            )
        except Exception as exc:
            st.error(f"无法预览 {upload.name}：{exc}")

st.header("2. 影像预处理")
with st.expander("调整预处理参数", expanded=True):
    c1, c2, c3 = st.columns(3)
    highlight = c1.slider("雪面高光压缩", 0.0, 0.8, 0.15, 0.05)
    clahe = c2.slider("局部对比度（CLAHE）", 0.0, 6.0, 1.5, 0.1)
    gamma = c3.slider("Gamma 亮度", 0.4, 2.5, 1.0, 0.05)
    c1, c2 = st.columns(2)
    denoise = c1.slider("双边降噪强度", 0.0, 60.0, 0.0, 2.0)
    sharpen = c2.slider("轻度锐化", 0.0, 1.0, 0.0, 0.05)
    max_edge = st.select_slider("输出最长边", options=[0, 1100, 1600, 2000, 2400, 3200], value=2400,
                                format_func=lambda value: "保留原尺寸" if value == 0 else f"{value} px")

prep = PreprocessConfig(
    highlight=highlight,
    clahe=clahe,
    gamma=gamma,
    denoise=denoise,
    sharpen=sharpen,
    max_edge=max_edge,
)

if st.button("生成预处理结果", type="primary", disabled=not uploads):
    if len(uploads) < 2:
        st.error("多视图重建至少需要两张照片。")
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="glacier_streamlit_"))
        image_dir = work_dir / "images"
        image_dir.mkdir(parents=True)
        previews: list[tuple[str, bytes]] = []
        progress = st.progress(0, text="正在处理照片…")
        try:
            for index, upload in enumerate(uploads):
                source = work_dir / f"source_{index:04d}{Path(upload.name).suffix.lower()}"
                source.write_bytes(upload.getvalue())
                rgb = preprocess(read_rgb(source), prep)
                target = image_dir / f"{index:06d}.png"
                write_png(target, rgb)
                previews.append((upload.name, target.read_bytes()))
                progress.progress((index + 1) / len(uploads), text=f"已处理 {index + 1}/{len(uploads)} 张")
            st.session_state["web_work_dir"] = str(work_dir)
            st.session_state["web_image_dir"] = str(image_dir)
            st.session_state["web_previews"] = previews
            st.session_state["web_prep_snapshot"] = prep.__dict__
            st.success(f"预处理完成：{len(previews)} 张。原始上传文件未被修改。")
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            st.exception(exc)

previews = st.session_state.get("web_previews", [])
if previews:
    st.subheader("预处理预览")
    cols = st.columns(min(3, len(previews)))
    for index, (name, content) in enumerate(previews):
        cols[index % len(cols)].image(content, caption=name, use_container_width=True)
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, (_, content) in enumerate(previews):
            archive.writestr(f"images/{index:06d}.png", content)
    st.download_button("下载预处理照片 ZIP", zip_buffer.getvalue(), "glacier_processed_images.zip", "application/zip")

st.header("3. 运行重建")
ready_methods = [method for method, label in (("A", "VGGT"), ("B", "COLMAP + OpenMVS"))
                 if method_status[method] == label]
if not ready_methods:
    st.warning("当前服务器没有检测到可运行的重建工具。照片上传、预处理和模型预览仍可使用；云端部署时可连接另行配置的 Linux 重建后端。")
else:
    method_labels = {"A": "VGGT", "B": "COLMAP + OpenMVS"}
    selected_labels = st.multiselect(
        "选择可用的重建路径",
        options=[method_labels[method] for method in ready_methods],
        default=[method_labels[method] for method in ready_methods],
    )
    label_to_method = {value: key for key, value in method_labels.items()}
    selected_methods = [label_to_method[label] for label in selected_labels]
    can_run = bool(st.session_state.get("web_image_dir")) and bool(selected_methods)
    if st.button("开始重建", type="primary", disabled=not can_run):
        run_dir = Path(tempfile.mkdtemp(prefix="glacier_web_run_"))
        status = st.status("重建任务已启动", expanded=True)
        progress = st.progress(0)
        log_area = st.empty()
        messages: list[str] = []
        results: dict[str, str] = {}
        stage_state = {"method_index": 0}

        def emit(message: str) -> None:
            messages.append(message)
            log_area.code("\n".join(messages[-80:]), language="text")

        def stage(done: int, total: int, text: str) -> None:
            fraction = done / max(total, 1)
            overall = (stage_state["method_index"] + fraction) / max(len(selected_methods), 1)
            progress.progress(min(1.0, max(0.0, overall)), text=text)
            status.write(text)

        try:
            for method in selected_methods:
                output_dir = run_dir / method
                output_dir.mkdir(parents=True, exist_ok=True)
                if method == "A":
                    model_path = run_vggt(
                        Path(st.session_state["web_image_dir"]), output_dir, config,
                        emit, stage, lambda: None,
                    )
                else:
                    model_path = run_colmap_openmvs(
                        Path(st.session_state["web_image_dir"]), output_dir, config,
                        emit, stage, lambda: None,
                    )
                results[method] = str(model_path)
                stage_state["method_index"] += 1
            st.session_state["web_results"] = results
            st.session_state["web_run_dir"] = str(run_dir)
            progress.progress(1.0, text="重建完成")
            status.update(label="重建完成", state="complete")
            st.success("所选重建路径已完成。")
        except Exception as exc:
            status.update(label="重建失败", state="error")
            st.error(f"重建失败：{exc}")
            if run_dir.exists():
                st.session_state["web_run_dir"] = str(run_dir)

results = st.session_state.get("web_results", {})
if results:
    st.header("4. 模型结果")
    for method, model_path in results.items():
        path = Path(model_path)
        if path.is_file():
            with st.expander(f"路径 {method} · {path.name}", expanded=True):
                st.caption(f"模型文件：{path.name}")
                st.download_button(
                    f"下载模型 {path.suffix.upper()}", path.read_bytes(), path.name,
                    key=f"download_{method}_{path.name}",
                )
                st.write("可在下方三维模型查看器中预览。")

st.header("三维模型查看器")
model_bytes = None
model_name = ""
if results:
    model_options = {f"路径 {method} · {Path(path).name}": path for method, path in results.items()}
    selected_model = st.selectbox("选择已重建模型", list(model_options))
    selected_path = Path(model_options[selected_model])
    if selected_path.is_file():
        model_bytes = selected_path.read_bytes()
        model_name = selected_path.name
model_upload = st.file_uploader("也可以上传 OBJ 或 PLY 模型预览", type=["obj", "ply"], key="model_upload")
if model_upload is not None:
    model_bytes = model_upload.getvalue()
    model_name = model_upload.name

if model_bytes and model_name:
    try:
        file_type = Path(model_name).suffix.lower().lstrip(".")
        loaded = trimesh.load(file_obj=BytesIO(model_bytes), file_type=file_type, force="scene", process=False)
        geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        meshes = [geometry for geometry in geometries if isinstance(geometry, trimesh.Trimesh) and len(geometry.vertices)]
        point_clouds = [geometry for geometry in geometries if isinstance(geometry, trimesh.points.PointCloud) and len(geometry.vertices)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            face_step = max(1, len(faces) // 120000)
            faces = faces[::face_step]
            colors = None
            try:
                vertex_colors = np.asarray(mesh.visual.vertex_colors)
                if len(vertex_colors) == len(vertices):
                    colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in vertex_colors[:, :3]]
            except Exception:
                pass
            fig = go.Figure(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                vertexcolor=colors,
                color="#b9d5e8" if colors is None else None,
                opacity=1.0, flatshading=False,
            ))
            geometry_summary = f"{len(vertices):,} 个顶点 · {len(mesh.faces):,} 个三角面"
        elif point_clouds:
            cloud = point_clouds[0]
            vertices = np.asarray(cloud.vertices)
            step = max(1, len(vertices) // 100000)
            vertices = vertices[::step]
            marker = {"size": 2, "color": "#d9e8f2"}
            if cloud.colors is not None and len(cloud.colors):
                marker = {"size": 2, "color": cloud.colors[::step, :3], "opacity": 0.9}
            fig = go.Figure(go.Scatter3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                mode="markers", marker=marker,
            ))
            geometry_summary = f"{len(cloud.vertices):,} 个点（预览最多抽样显示 100,000 点）"
        else:
            raise ValueError("文件中没有可预览的三角网格或点云几何。")
        fig.update_layout(
            height=650, margin=dict(l=0, r=0, t=20, b=0),
            scene=dict(aspectmode="data", bgcolor="#0d1420",
                       xaxis=dict(title="X", backgroundcolor="#0d1420", gridcolor="#334155"),
                       yaxis=dict(title="Y", backgroundcolor="#0d1420", gridcolor="#334155"),
                       zaxis=dict(title="Z", backgroundcolor="#0d1420", gridcolor="#334155")),
            paper_bgcolor="#0d1420", font=dict(color="#e5eef5"),
        )
        st.caption(f"{model_name} · {geometry_summary}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"模型预览失败：{exc}。OBJ 纹理文件如需显示，请与 MTL 和贴图一并提供；当前预览侧重几何形态。")
else:
    st.caption("上传 OBJ/PLY，或先运行可用的重建路径，即可预览模型。")

st.caption("原始照片和模型仅在当前 Streamlit 会话/服务器临时目录中处理。托管服务重启后，临时结果可能被清除。")
