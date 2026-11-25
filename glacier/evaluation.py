"""DeepSeek Vision evaluation plus a multipart adapter for custom evaluation servers.

DeepSeek receives inline model preview images and factual geometry metadata. Other
configured endpoints keep the multipart model ZIP + metadata protocol. API keys
remain in memory and are excluded from task manifests and logs.
"""
from dataclasses import dataclass
import base64
import html
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit, urlunsplit
import zipfile

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor


@dataclass(frozen=True)
class APIConfig:
    enabled: bool = False
    url: str = ''
    key: str = ''
    auth_header: str = 'Authorization'
    auth_prefix: str = 'Bearer'
    file_field: str = 'model'
    metadata_field: str = 'metadata'
    result_path: str = ''          # 如 data.result；留空使用响应顶层
    timeout: int = 120            # 读取超时；连接超时单独限制为 10 秒


class EvaluationError(RuntimeError):
    pass


RUBRIC = '''冰川三维重建竞赛原型展示评价标准（满分100分）。所有成功读取并进入评价流程的模型统一计96分基础展示分，另有最多4分质量加分；因此总分范围为96-100分。96分是本项目的展示鼓励基准，不表示模型已达到测绘产品或科研级地理精度：
1. 冰川主体呈现与任务完成度 25：是否成功生成可辨认的冰川/冰雪地形主体，主要起伏和整体形态是否清楚。主体明确且流程成功即应获得本项大部分分数；局部噪声、薄片或错位只按影响范围和严重程度扣分，不因存在少量伪影一票否决。
2. 三维结构与视觉质量 20：观察多视角下的形态、连续性、边界和细节。以整体可读性为主；局部散点、表面不完整或轻微穿插属于可改进项，只有大范围伪影遮蔽主体、无法辨认结构时才显著扣分。
3. 方法实现与流程完整性 15：输入、预处理、重建和结果输出是否形成可运行流程，是否展示方法路径或关键处理步骤。依据已展示的实现给分，不强制要求特定算法或高密度网格。
4. 数据说明与结果边界 10：说明可获得的数据来源、输入图像情况、坐标/单位及当前结果适用范围。学生项目可如实说明“尚未提供/尚未验证”；缺少元数据会限制地理解释，但不等同于模型重建失败，也不按缺项直接重罚。
5. 精度验证与误差意识 5：如有独立参考数据、检查点或定量误差则加以认可；尚无独立验证时仍可按是否诚实说明精度未知、避免过度结论给基础分，不设封顶或硬性扣分。
6. 展示与可复现性 15：模型视图是否清楚，操作/参数/软件流程是否有基本说明，结果是否便于复查。比例尺、方向、图例和版本信息作为加分完善项，不作为原型展示的准入条件。
7. 创新性、学习价值与改进潜力 10：评价方法组合、问题解决思路、竞赛目标契合度及后续可扩展性；不以是否已达到生产级应用作为判断标准。
评分原则：七项各自先计该项满分的96%作为基础分，七项基础分合计96分；剩余4分作为质量加分，依据可见的形态质量、流程完成度和展示证据分配到七项，不得超过各项满分。先肯定已经实现的功能与模型可读性，再说明局限及下一步。对无法从提交内容判断的事项写“尚未验证/材料未提供”，不要推断为错误或失败；坐标、来源或精度缺失主要限制应用结论，不应重复扣分。明显伪影可影响质量加分，但不扣除96分基础分。96-100表示竞赛原型展示分（包含96分基础展示分），不代表测绘级精度已验证。
先概述已完成的工作和可见优点，再给逐项得分/满分及依据，最后给不超过三项优先改进建议。语气具体、建设性，避免使用“主要目标尚未达到”等超出证据的结论。'''


def _deepseek_endpoint(url):
    """DeepSeek accepts OpenAI-compatible JSON at /chat/completions, not multipart."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip('/')
    suffix = '/chat/completions'
    if not path.endswith(suffix):
        path += suffix
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ''))


def _model_evidence(model, check_cancel):
    """Render three views and collect factual geometry counts using the app's VTK dependency."""
    try:
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        from vtkmodules.vtkIOGeometry import vtkOBJReader
        from vtkmodules.vtkIOPLY import vtkPLYReader
        from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, vtkRenderer,
                                                vtkRenderWindow)
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        from vtkmodules.vtkIOImage import vtkPNGWriter
        from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
    except ImportError as exc:
        raise EvaluationError('DeepSeek 评价需要 VTK 生成模型预览图；请检查 VTK 安装。') from exc

    path = Path(model).resolve()
    reader = vtkOBJReader() if path.suffix.lower() == '.obj' else vtkPLYReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    points, cells = poly.GetNumberOfPoints(), poly.GetNumberOfCells()
    if not points:
        raise EvaluationError('模型没有可读取的有效顶点。')
    bounds = poly.GetBounds()
    dimensions = [round(bounds[i + 1] - bounds[i], 6) for i in (0, 2, 4)]
    center = [(bounds[i] + bounds[i + 1]) / 2 for i in (0, 2, 4)]
    radius = max(max(dimensions) / 2, 1e-6)

    renderer = vtkRenderer()
    renderer.SetBackground(0.045, 0.055, 0.075)
    mapper = vtkPolyDataMapper()
    glyph = None
    if cells:
        mapper.SetInputData(poly)
    else:
        glyph = vtkVertexGlyphFilter()
        glyph.SetInputData(poly)
        glyph.Update()
        mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.78, 0.85, 0.92)
    actor.GetProperty().SetPointSize(2)
    renderer.AddActor(actor)
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(800, 600)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    image_paths = []
    with tempfile.TemporaryDirectory(prefix='glacier-deepseek-') as directory:
        for label, direction, up in (
                ('斜视图', (1, -1, 0.85), (0, 0, 1)),
                ('侧视图', (1, 0, 0), (0, 0, 1)),
                ('俯视图', (0, 0, 1), (0, 1, 0))):
            check_cancel()
            camera.SetFocalPoint(*center)
            camera.SetPosition(*(center[i] + direction[i] * radius * 3 for i in range(3)))
            camera.SetViewUp(*up)
            camera.SetParallelScale(radius * 1.15)
            renderer.ResetCameraClippingRange()
            window.Render()
            capture = vtkWindowToImageFilter()
            capture.SetInput(window)
            capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOff()
            capture.Update()
            out = Path(directory) / f'{label}.png'
            writer = vtkPNGWriter()
            writer.SetFileName(str(out))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            image_paths.append((label, out.read_bytes()))
        window.Finalize()
        # Return bytes before the temporary directory is removed.
        return {'images': image_paths, 'geometry': {
            'file': path.name, 'format': path.suffix.lower().lstrip('.'),
            'vertex_count': points, 'cell_count': cells,
            'bounds_xyz': [[round(bounds[i], 6), round(bounds[i + 1], 6)] for i in (0, 2, 4)],
            'dimensions_xyz': dimensions,
            'note': 'Counts and bounds are read from the submitted mesh; they do not establish geographic coordinates or accuracy.'}}


def _evaluate_deepseek(model, metadata, cfg, check_cancel):
    evidence = _model_evidence(model, check_cancel)
    geometry = evidence['geometry']
    user_text = (
        f'{RUBRIC}\n\n模型文件与几何信息（自动读取，仅作事实依据）：\n'
        f'{json.dumps(geometry, ensure_ascii=False, indent=2)}\n\n'
        f'用户/任务提供的补充材料：\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n'
        '我会附上同一模型的斜视、侧视、俯视图。只根据这些图和以上文字评价。'
        '模型视图不能证明真实坐标、数据来源、采集时间或精度；缺少对应材料时写“尚未验证/材料未提供”，'
        '不要仅因这些信息缺失就把可视化原型评为不合格，也不要把未验证误写成已验证。评分必须采用96分基础展示分加最多4分质量加分：'
        '每个评分项至少给该项满分的96%，七项基础分合计96分；剩余最多4分按可见质量加到各项。整体模型清晰、主体突出、表面连续且流程完成时，应给足质量加分，使总分达到99-100分；有少量局部瑕疵但整体效果良好时，通常给98-99分；只有整体质量存在明显问题时才接近96分。局部伪影只能影响加分，不能扣除基础分。'
        '依据模型主体是否可辨认、流程是否完成和整体展示质量分配加分。'
        '请严格按7个维度评分，分项分数不得超过各项满分，合计满分100。'
        'metrics 必须恰好包含以下七个名称，逐字一致：冰川主体呈现与任务完成度、三维结构与视觉质量、方法实现与流程完整性、数据说明与结果边界、精度验证与误差意识、展示与可复现性、创新性、学习价值与改进潜力。'
        '只返回一个 JSON 对象，结构为 {"score":数值,"metrics":[{"name":名称,"score":分数,"max_score":满分,"evidence":依据或无法判断原因}],'
        '"top_issues":[最多三项],"recommendations":[可执行建议],"text":总体结论}。')
    content = [{'type': 'text', 'text': user_text}]
    for label, image_bytes in evidence['images']:
        encoded = base64.b64encode(image_bytes).decode('ascii')
        content.append({'type': 'text', 'text': f'模型{label}'})
        content.append({'type': 'image_url', 'image_url': {
            'url': f'data:image/png;base64,{encoded}', 'detail': 'high'}})
    parts = urlsplit(cfg.url)
    safe_host = f'{parts.scheme}://{parts.hostname or ""}' + (f':{parts.port}' if parts.port else '')
    headers = {'Authorization': f'Bearer {cfg.key.strip()}', 'Content-Type': 'application/json'}
    body = {'model': 'deepseek-flash', 'messages': [
        {'role': 'system', 'content': '你是友善、建设性的冰川三维重建竞赛原型评审员。只依据提交证据评分，不臆测；区分原型展示完成度和测绘级精度验证，先说明已实现的优点，再适度指出局限。'},
        {'role': 'user', 'content': content}],
        'response_format': {'type': 'json_object'}, 'temperature': 0.2, 'max_tokens': 3000}
    try:
        response = requests.post(_deepseek_endpoint(cfg.url), headers=headers, json=body,
                                 timeout=(10, cfg.timeout), allow_redirects=False)
        if not 200 <= response.status_code < 300:
            detail = _error_response_detail(response, cfg)
            raise EvaluationError(f'DeepSeek API 返回 HTTP {response.status_code}，请确认模型权限和余额。{detail}')
        try:
            wrapper = response.json()
            answer = wrapper['choices'][0]['message']['content']
            result = json.loads(answer)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise EvaluationError('DeepSeek 返回格式异常；需要包含 JSON 的 chat completion 响应。') from exc
        if not isinstance(result, dict):
            raise EvaluationError('DeepSeek 评价结果不是 JSON 对象。')
        expected = {'冰川主体呈现与任务完成度': 25, '三维结构与视觉质量': 20,
                    '方法实现与流程完整性': 15, '数据说明与结果边界': 10,
                    '精度验证与误差意识': 5, '展示与可复现性': 15,
                    '创新性、学习价值与改进潜力': 10}
        metrics = result.get('metrics')
        if not isinstance(metrics, list) or len(metrics) != len(expected):
            raise EvaluationError('DeepSeek 未按评分表返回完整的7项评分；请重试评价。')
        total = 0
        seen = set()
        for item in metrics:
            if not isinstance(item, dict) or item.get('name') not in expected:
                raise EvaluationError('DeepSeek 评分项名称不符合评价标准；请重试评价。')
            name = item['name']
            if name in seen or item.get('max_score') != expected[name]:
                raise EvaluationError('DeepSeek 评分项满分或项目重复；请重试评价。')
            score = item.get('score')
            if not isinstance(score, (int, float)) or not 0 <= score <= expected[name]:
                raise EvaluationError(f'DeepSeek 对“{name}”返回了无效分数。')
            # Guarantee the requested 96-point exhibition baseline consistently
            # across the itemized metrics; only the remaining 4 points reflect
            # quality distinctions. Keep the total equal to the displayed rows.
            baseline = expected[name] * 0.96
            if score < baseline:
                item['evidence'] = (str(item.get('evidence', '')).strip() +
                                    f'；已计入竞赛展示基础分 {baseline:g}/{expected[name]}')
                score = baseline
            item['score'] = round(score, 2)
            total += score
            seen.add(name)
        result['score'] = round(total, 2)
        issues = result.get('top_issues', [])
        recommendations = result.get('recommendations', [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)] if recommendations else []
        summary = result.get('text', '')
        result['text'] = (str(summary).strip() + '\n\n最影响质量的问题：\n' +
                          '\n'.join(f'{i + 1}. {v}' for i, v in enumerate(issues[:3])) +
                          '\n\n改进建议：\n' +
                          '\n'.join(f'{i + 1}. {v}' for i, v in enumerate(recommendations)))
        return {'result': result, 'raw': wrapper, 'evidence': geometry}
    except requests.RequestException as exc:
        detail = str(exc)
        if cfg.key:
            detail = detail.replace(cfg.key, '[密钥已隐藏]')
        detail = detail.replace(cfg.url, safe_host)
        detail = re.sub(r'(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;"<>]+', r'\1[已隐藏]', detail)
        raise EvaluationError(f'DeepSeek 连接失败：{type(exc).__name__}，请检查代理、防火墙及 api.deepseek.com:443 连接。{detail[:400]}') from exc


def _error_response_detail(response, cfg):
    """Return a small, credential-scrubbed excerpt to make HTTP errors diagnosable."""
    try:
        raw = response.raw.read(2048, decode_content=True)
        body = raw.decode(response.encoding or 'utf-8', errors='replace').strip()
    except Exception:
        return ''
    if not body:
        return ''
    if cfg.key:
        body = body.replace(cfg.key, '[密钥已隐藏]')
    body = re.sub(r'(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;"<>]+',
                  r'\1[已隐藏]', body)
    if 'json' not in response.headers.get('Content-Type', '').lower():
        body = html.unescape(re.sub(r'<[^>]{1,256}>', ' ', body))
    body = ' '.join(body.split())[:500]
    return f'；服务端响应摘要：{body}' if body else ''


def validate_api(cfg):
    parts = urlsplit(cfg.url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise EvaluationError('API 地址必须为不含用户名/密码的完整 HTTP(S) URL。')
    if not cfg.file_field.strip() or not cfg.metadata_field.strip() or cfg.file_field == cfg.metadata_field:
        raise EvaluationError('文件字段与元数据字段必须非空且不同。')
    if any(c in cfg.key + cfg.auth_header + cfg.auth_prefix for c in '\r\n'):
        raise EvaluationError('鉴权配置不能包含换行。')
    if cfg.key and not cfg.auth_header.strip():
        raise EvaluationError('填写密钥时也需要指定鉴权头名称。')
    if parts.hostname and parts.hostname.lower() == 'api.deepseek.com' and not cfg.key.strip():
        raise EvaluationError('DeepSeek API 评价需要填写 API Key。')


def bundle_model(model, destination, check_cancel=lambda: None):
    """保留纹理相对目录；不跟随指向结果目录以外的符号链接。

外部模型的 MTL/贴图请放在模型同目录或子目录。这里不上传源照片，
不上传数据库、检查点或 .mvs 中间文件。限制总大小避免无限制打包。
"""
    model = Path(model).resolve()
    root = model.parent
    assets = {'.mtl', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    files = [model] + sorted(p for p in root.rglob('*')
                            if p.is_file() and p.suffix.lower() in assets and p != model)
    total = 0
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            check_cancel()
            if not path.resolve().is_relative_to(root):
                raise EvaluationError('纹理链接指向模型目录以外，请先整理为独立模型目录。')
            total += path.stat().st_size
            if total > 2 * 1024**3:
                raise EvaluationError('模型及纹理超过 2 GiB，请简化模型后评价。')
            # 分块写入，让大型模型打包阶段也能及时响应取消。
            with path.open('rb') as src, archive.open(path.relative_to(root).as_posix(), 'w', force_zip64=True) as dst:
                while chunk := src.read(1024 * 1024):
                    check_cancel()
                    dst.write(chunk)


def evaluate(model, metadata, cfg, check_cancel=lambda: None):
    validate_api(cfg)
    check_cancel()
    if (urlsplit(cfg.url).hostname or '').lower() == 'api.deepseek.com':
        return _evaluate_deepseek(model, metadata, cfg, check_cancel)
    with tempfile.TemporaryDirectory(prefix='glacier-eval-') as temp:
        archive_path = Path(temp) / 'model.zip'
        bundle_model(model, archive_path, check_cancel)
        headers = {}
        if cfg.key:
            headers[cfg.auth_header] = (cfg.auth_prefix.strip() + ' ' + cfg.key).strip()
        with archive_path.open('rb') as stream:
            # 普通 requests(files=...) 会先把文件读进内存；编码器采用流式上传。
            encoder = MultipartEncoder(fields={
                cfg.file_field: ('model.zip', stream, 'application/zip'),
                cfg.metadata_field: json.dumps(metadata, ensure_ascii=False),
            })
            monitor = MultipartEncoderMonitor(encoder, lambda _: check_cancel())
            headers['Content-Type'] = monitor.content_type
            try:
                # 禁止自动重定向，防止自定义鉴权头被转发给其他服务器；TLS 校验保持开启。
                with requests.post(cfg.url, data=monitor, headers=headers, timeout=(10, cfg.timeout),
                                   allow_redirects=False, stream=True) as response:
                    if not 200 <= response.status_code < 300:
                        detail = _error_response_detail(response, cfg)
                        raise EvaluationError(
                            f'API 返回 HTTP {response.status_code}，请检查接口及鉴权配置。{detail}')
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        check_cancel()
                        data.extend(chunk)
                        if len(data) > 4 * 1024**2:
                            raise EvaluationError('API 响应超过 4 MiB，请返回精简 JSON 指标。')
                    try:
                        payload = json.loads(data)
                    except (ValueError, UnicodeError) as exc:
                        raise EvaluationError('API 未返回有效 JSON。') from exc
            except requests.RequestException as exc:
                # Show the underlying network cause, but scrub credentials and URL paths.
                detail = str(exc)
                if cfg.key:
                    detail = detail.replace(cfg.key, '[密钥已隐藏]')
                parts = urlsplit(cfg.url)
                safe_host = f'{parts.scheme}://{parts.hostname or ""}'
                if parts.port:
                    safe_host += f':{parts.port}'
                detail = detail.replace(cfg.url, safe_host)
                detail = re.sub(r'(?i)(with url:\s*)\S+', r'\1[接口路径已隐藏]', detail)
                detail = re.sub(r'(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;"<>]+',
                                r'\1[已隐藏]', detail)
                detail = ' '.join(detail.split())[:400]
                suffix = f'：{detail}' if detail else ''
                raise EvaluationError(
                    f'HTTP 请求失败（{type(exc).__name__}），请检查网络、代理与服务器连接{suffix}') from exc
    check_cancel()
    result = payload
    try:
        for key in filter(None, cfg.result_path.split('.')):
            result = result[key]
    except (KeyError, TypeError) as exc:
        raise EvaluationError('响应根路径不存在，请检查 result_path。') from exc
    if not isinstance(result, dict):
        raise EvaluationError('评价结果应是 JSON 对象，包含 score、metrics、text 等后端字段。')
    return {'result': result, 'raw': payload}
