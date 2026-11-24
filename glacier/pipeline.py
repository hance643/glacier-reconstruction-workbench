"""算法外部调度；GUI 环境不导入算法库，命令使用参数数组和 shell=False。"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import threading
import psutil


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineConfig:
    vggt_python: str = ''
    vggt_repo: str = ''
    vggt_args: str = '[]'
    colmap: str = 'colmap'
    openmvs_bin: str = ''
    colmap_extras: str = '{}'
    openmvs_extras: str = '{}'


def argument_list(value):
    args = json.loads(value)
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        raise ValueError('附加参数必须是 JSON 字符串数组，例如 ["--use_ba"]。')
    return args


def extra_map(value):
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise ValueError('阶段附加参数必须是 JSON 对象。')
    for key, args in obj.items():
        argument_list(json.dumps(args))
        reserved = {'--database_path', '--image_path', '--input_path', '--output_path', '--output_type',
                    '-i', '-o', '-w', '--input-file', '--output-file', '--working-folder', '--export-type'}
        if any(a.split('=')[0] in reserved for a in args):
            raise ValueError(f'{key} 的附加参数不可覆盖输入、输出或工作目录。')
    return obj


def executable(value):
    value = os.path.expandvars(str(value).strip())
    candidate = Path(value).expanduser()
    resolved = str(candidate.resolve()) if candidate.is_file() else shutil.which(value)
    if not resolved:
        raise FileNotFoundError(f'找不到可执行文件：{value}')
    if Path(resolved).suffix.lower() in {'.bat', '.cmd'}:
        raise ValueError('请选择真正的 .exe 文件；本框架不通过 shell 执行 BAT/CMD。')
    return resolved


def mvs_executable(cfg, name):
    base = Path(cfg.openmvs_bin).expanduser() if cfg.openmvs_bin.strip() else None
    return executable(str(base / (name + ('.exe' if os.name == 'nt' else ''))) if base else name)


def validate_tools(methods, cfg):
    """先验证工具与参数，再处理照片。GPU、权重和版本兼容性仍由外部工具检查。"""
    if 'A' in methods:
        if not cfg.vggt_python.strip():
            raise ValueError('VGGT Python 尚未配置或未自动检测到可用环境。请运行“自动检测工具”，选择包含 VGGT 依赖的 Python；也可检查 VGGT/.venv/pyvenv.cfg 中的基础 Python 是否仍存在。')
        if not cfg.vggt_repo.strip():
            raise ValueError('VGGT 仓库目录尚未配置，请选择包含 demo_colmap.py 的仓库目录。')
        executable(cfg.vggt_python)
        if not (Path(cfg.vggt_repo).expanduser() / 'demo_colmap.py').is_file():
            raise ValueError('VGGT 仓库目录中没有 demo_colmap.py。')
        if any(a.split('=')[0] == '--scene_dir' for a in argument_list(cfg.vggt_args)):
            raise ValueError('不要在附加参数中覆盖 --scene_dir。')
    if 'B' in methods:
        executable(cfg.colmap)
        for name in ['InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh']:
            mvs_executable(cfg, name)
        for raw, allowed in [(cfg.colmap_extras, {'feature_extractor', 'exhaustive_matcher', 'mapper', 'image_undistorter'}),
                             (cfg.openmvs_extras, {'InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh'})]:
            unknown = set(extra_map(raw)) - allowed
            if unknown:
                raise ValueError(f'未知调度阶段：{", ".join(sorted(unknown))}')


def terminate_tree(process):
    """取消时终止整个算法进程树，防止 GPU 子进程继续运行。"""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=2)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def run_process(argv, cwd, emit, check_cancel):
    """日志读取与轮询分离；工具完全不输出日志时，仍可及时取消。"""
    check_cancel()
    emit('命令：' + subprocess.list2cmdline([str(a) for a in argv]))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    env['PATH'] = str(Path(argv[0]).parent) + os.pathsep + env.get('PATH', '')
    env.pop('QT_PLUGIN_PATH', None)
    env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=flags, shell=False)
    messages = queue.Queue(maxsize=1000)
    stop_reader = threading.Event()

    def read_lines():
        try:
            while raw := process.stdout.readline(16384):
                message = raw.decode('utf-8', errors='replace').rstrip()
                while not stop_reader.is_set():
                    try:
                        messages.put(message, timeout=0.1)
                        break
                    except queue.Full:
                        pass
                if stop_reader.is_set():
                    break
        finally:
            process.stdout.close()

    reader = threading.Thread(target=read_lines, daemon=True)
    reader.start()
    try:
        while process.poll() is None or reader.is_alive() or not messages.empty():
            check_cancel()
            try:
                emit(messages.get(timeout=0.1))
            except queue.Empty:
                pass
        code = process.wait()
        if code:
            raise RuntimeError(f'{Path(argv[0]).name} 退出码 {code}；请查看完整任务日志。')
    except BaseException:
        stop_reader.set()
        terminate_tree(process)
        raise
    finally:
        stop_reader.set()
        reader.join(timeout=2)


def registered_count(folder):
    """COLMAP images.bin 首个 uint64 为注册图像数，用于选择最大连通模型。"""
    path = folder / 'images.bin'
    if path.is_file():
        with path.open('rb') as stream:
            header = stream.read(8)
        return struct.unpack('<Q', header)[0] if len(header) == 8 else 0
    path = folder / 'images.txt'
    if path.is_file():
        return len([line for line in path.read_text(encoding='utf-8').splitlines()
                    if not line.startswith('#')]) // 2
    return 0


def run_vggt(images, output_dir, cfg, emit, stage, check_cancel):
    scene = output_dir / 'scene'
    scene_images = scene / 'images'
    scene_images.mkdir(parents=True)
    for path in sorted(images.glob('*.png')):
        check_cancel()
        shutil.copy2(path, scene_images / path.name)
    repo = Path(cfg.vggt_repo).expanduser().resolve()
    stage(0, 1, 'VGGT 推理 / COLMAP 导出')
    run_process([executable(cfg.vggt_python), repo / 'demo_colmap.py', '--scene_dir', scene,
                 *argument_list(cfg.vggt_args)], repo, emit, check_cancel)
    candidate = scene / 'sparse' / 'points.ply'
    if not candidate.is_file():
        candidates = sorted(scene.rglob('*.ply'))
        if len(candidates) != 1:
            raise RuntimeError('VGGT 已退出，但没有唯一可识别的 PLY 输出，请检查所安装版本及日志。')
        candidate = candidates[0]
    model_dir = output_dir / 'model'
    model_dir.mkdir()
    model = model_dir / 'vggt_points.ply'
    shutil.copy2(candidate, model)
    stage(1, 1, 'VGGT 点云导出完成')
    return model


def run_colmap_openmvs(images, output_dir, cfg, emit, stage, check_cancel):
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse, dense, model_dir = (output_dir / name for name in ['sparse', 'dense', 'model'])
    for path in [sparse, dense, model_dir]:
        path.mkdir()
    db = output_dir / 'database.db'
    colmap = executable(cfg.colmap)
    ce, me = extra_map(cfg.colmap_extras), extra_map(cfg.openmvs_extras)
    count = 0

    def step(name, argv):
        nonlocal count
        stage(count, 9, name)
        run_process(argv, output_dir, emit, check_cancel)
        count += 1
        stage(count, 9, name + ' 完成')

    step('特征提取', [colmap, 'feature_extractor', '--database_path', db, '--image_path', images,
                     *ce.get('feature_extractor', [])])
    step('全配对特征匹配', [colmap, 'exhaustive_matcher', '--database_path', db, *ce.get('exhaustive_matcher', [])])
    step('稀疏重建', [colmap, 'mapper', '--database_path', db, '--image_path', images,
                     '--output_path', sparse, *ce.get('mapper', [])])
    components = [p for p in sparse.iterdir() if p.is_dir() and registered_count(p) >= 2]
    if not components:
        raise RuntimeError('COLMAP 未生成至少两张图像的连通重建，请检查重叠率、清晰度及纹理。')
    selected = max(components, key=registered_count)
    emit(f'采用最大连通模型 {selected.name}，注册图像 {registered_count(selected)} 张。')
    step('图像去畸变', [colmap, 'image_undistorter', '--image_path', images, '--input_path', selected,
                       '--output_path', dense, '--output_type', 'COLMAP', *ce.get('image_undistorter', [])])
    scene, dense_scene, mesh, refined = [output_dir / name for name in
                                       ['scene.mvs', 'scene_dense.mvs', 'scene_mesh.mvs', 'scene_refined.mvs']]
    commands = [
        ('InterfaceCOLMAP', ['-i', dense, '-o', scene]),
        ('DensifyPointCloud', ['-i', scene, '-o', dense_scene]),
        ('ReconstructMesh', ['-i', dense_scene, '-o', mesh]),
        ('RefineMesh', ['-i', mesh, '-o', refined]),
        ('TextureMesh', ['-i', refined, '-o', model_dir / 'glacier.mvs', '--export-type', 'obj']),
    ]
    for name, args in commands:
        step(name, [mvs_executable(cfg, name), *args, *me.get(name, [])])
    expected = model_dir / 'glacier.obj'
    if not expected.is_file():
        raise RuntimeError('未找到 TextureMesh 预期的 model/glacier.obj，请检查 OpenMVS 版本及日志。')
    return expected
