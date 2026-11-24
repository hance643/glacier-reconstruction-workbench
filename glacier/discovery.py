"""有界、只读的本地工具发现；不下载工具、不运行重建、不扫描整个磁盘。

查找顺序：当前配置 / 环境变量 / PATH / 常见目录和项目附近目录。
仅当候选唯一时由界面自动填入；存在多个安装版本时保留选择权。
Python 解释器只执行固定的依赖存在性探测，不运行仓库中的脚本。
"""
from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

KEYS = ('vggt_python', 'vggt_repo', 'colmap', 'openmvs_bin')
MVS_NAMES = ('InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh')
SKIP = {'.git', '.svn', 'node_modules', '__pycache__', 'site-packages', '.cache',
        'images', 'dense', 'sparse', 'outputs', 'logs', 'data', 'datasets', '冰川数据',
        'windows', '$recycle.bin', 'system volume information'}


def canonical(path):
    return str(Path(os.path.expandvars(str(path))).expanduser().resolve())


def is_tool_path(key, value):
    """检测保存路径是否仍存在；不把路径存在误称为算法已可运行。"""
    if not value:
        return False
    path = Path(os.path.expandvars(str(value))).expanduser()
    if key == 'vggt_repo':
        return (path / 'demo_colmap.py').is_file()
    if key == 'openmvs_bin':
        suffix = '.exe' if os.name == 'nt' else ''
        return all((path / (name + suffix)).is_file() for name in MVS_NAMES)
    return path.is_file() and path.suffix.lower() not in {'.bat', '.cmd'}


def standard_roots(project, environ=None):
    env = os.environ if environ is None else environ
    home = Path.home()
    roots = [Path(project), Path(project).parent]
    for name in ('Desktop', 'Downloads', 'Documents', 'source', 'repos', 'tools'):
        roots.append(home / name)
    for name in ('ProgramFiles', 'ProgramFiles(x86)', 'VGGT_HOME', 'COLMAP_HOME', 'OPENMVS_HOME'):
        if env.get(name):
            roots.append(Path(env[name]))
    if env.get('LOCALAPPDATA'):
        roots.append(Path(env['LOCALAPPDATA']) / 'Programs')
    # 只查看磁盘根目录中名称相关的安装文件夹，不进行全盘遍历。
    if os.name == 'nt':
        for drive in ('C:/', 'D:/', 'E:/', 'F:/'):
            try:
                for child in Path(drive).iterdir():
                    if child.is_dir() and any(token in child.name.lower() for token in
                        ('colmap', 'openmvs', 'vggt', 'python', 'conda', 'tools', 'software', 'program', 'app', '软件')):
                        roots.append(child)
            except OSError:
                pass
    return roots


def python_probe(path, check=lambda: None, timeout=5):
    """只验证核心模块是否存在；不加载权重、不初始化 GPU、不承诺 CUDA 可用。"""
    venv_config = Path(path).parent.parent / 'pyvenv.cfg'
    if venv_config.is_file():
        try:
            settings = {}
            for line in venv_config.read_text(encoding='utf-8', errors='replace').splitlines():
                key, separator, value = line.partition('=')
                if separator:
                    settings[key.strip().lower()] = value.strip()
            home = settings.get('home')
            if home:
                base = Path(os.path.expandvars(home)).expanduser()
                if not (base / 'python.exe').is_file():
                    return False, f'虚拟环境失效：pyvenv.cfg 指向的基础 Python 不存在（{base / "python.exe"}）。请用现有 Python 重建此虚拟环境。'
        except OSError as exc:
            return False, f'虚拟环境基础 Python 路径无法访问（{type(exc).__name__}）；请检查 pyvenv.cfg 中的 home 路径及权限'
    script = (
        "import importlib.util,json;"
        "names=['torch','torchvision','pycolmap','cv2','numpy','PIL'];"
        "print(json.dumps({n:importlib.util.find_spec(n) is not None for n in names}))"
    )
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = None
    try:
        check()
        proc = subprocess.Popen([str(path), '-I', '-c', script], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=flags)
        deadline = time.monotonic() + timeout
        while True:
            check()
            try:
                out, _ = proc.communicate(timeout=min(.15, max(.01, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    return False, '依赖检查超时'
        if proc.returncode:
            return False, '解释器未正常退出'
        modules = json.loads(out.decode('utf-8'))
        missing = [name for name, exists in modules.items() if not exists]
        return (not missing), ('核心依赖存在；权重/CUDA 未验证' if not missing else '缺少 ' + ', '.join(missing))
    except (OSError, ValueError, UnicodeError) as exc:
        return False, f'解释器启动失败（{type(exc).__name__}），检查虚拟环境基础 Python 路径'
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate()


def discover(project, current=None, report=lambda _: None, check=lambda: None,
             roots=None, environ=None, scan_seconds=12, max_dirs=12000, probe_limit=10):
    env = os.environ if environ is None else environ
    current = current or {}
    result = {key: [] for key in KEYS}
    notes, python_paths = [], []

    def add(key, path):
        try:
            value = canonical(path)
            if is_tool_path(key, value) and value not in result[key]:
                result[key].append(value)
        except (OSError, ValueError):
            pass

    def add_python(path):
        try:
            value = canonical(path)
            if is_tool_path('vggt_python', value) and value not in python_paths:
                python_paths.append(value)
        except (OSError, ValueError):
            pass

    for key in KEYS:
        value = current.get(key)
        if value:
            resolved = shutil.which(value, path=env.get('PATH', '')) or value
            (add_python(resolved) if key == 'vggt_python' else add(key, resolved))
    for name, key in [('VGGT_HOME', 'vggt_repo'), ('COLMAP_HOME', 'colmap'), ('OPENMVS_HOME', 'openmvs_bin')]:
        if env.get(name):
            base = Path(env[name])
            if key == 'colmap':
                for path in [base, base / 'colmap.exe', base / 'bin' / 'colmap.exe']:
                    add(key, path)
            else:
                add(key, base)
                if key == 'openmvs_bin':
                    add(key, base / 'bin')
    if env.get('VGGT_PYTHON'):
        add_python(env['VGGT_PYTHON'])
    for name, key in [('colmap', 'colmap'), ('InterfaceCOLMAP', 'openmvs_bin')]:
        value = shutil.which(name, path=env.get('PATH', ''))
        if value:
            add(key, Path(value).parent if key == 'openmvs_bin' else value)
    for name in ['python', 'python3']:
        value = shutil.which(name, path=env.get('PATH', ''))
        if value and 'windowsapps' not in value.lower():
            add_python(value)
    # GUI 自身可能没有 torch，但它也是候选之一，不武断当成 VGGT 环境。
    add_python(Path(sys.executable).with_name('python.exe') if os.name == 'nt' else sys.executable)
    home = Path.home()
    env_dirs = []
    try:
        env_dirs.extend(Path(line) for line in (home / '.conda' / 'environments.txt').read_text(encoding='utf-8').splitlines() if line.strip())
    except OSError:
        pass
    for name in ('miniconda3', 'anaconda3', 'miniforge3', 'mambaforge'):
        base = home / name
        env_dirs.append(base)
        try:
            env_dirs.extend(p for p in (base / 'envs').iterdir() if p.is_dir())
        except OSError:
            pass
    if env.get('CONDA_PREFIX'):
        env_dirs.append(Path(env['CONDA_PREFIX']))
    for base in env_dirs:
        for relative in ('python.exe', 'Scripts/python.exe', 'bin/python'):
            add_python(base / relative)

    pending = deque((Path(p), 0) for p in (standard_roots(project, env) if roots is None else roots))
    visited = set()
    deadline = time.monotonic() + scan_seconds
    limited = False
    while pending:
        check()
        if len(visited) >= max_dirs or time.monotonic() > deadline:
            limited = True
            break
        folder, depth = pending.popleft()
        try:
            normalized = canonical(folder)
            if normalized in visited or folder.is_symlink() or (hasattr(folder, 'is_junction') and folder.is_junction()):
                continue
            visited.add(normalized)
            with os.scandir(folder) as entries:
                files, children = {}, []
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        files[entry.name.lower()] = entry.name
                    elif entry.is_dir(follow_symlinks=False):
                        children.append(Path(entry.path))
            if 'demo_colmap.py' in files:
                add('vggt_repo', folder)
            for name in ('colmap.exe', 'colmap'):
                if name in files:
                    add('colmap', folder / files[name])
            if 'interfacecolmap.exe' in files or 'interfacecolmap' in files:
                add('openmvs_bin', folder)
            if 'python.exe' in files:
                add_python(folder / files['python.exe'])
            # 虚拟环境不递归进入 site-packages，仅直接查看解释器。
            for relative in ('.venv/Scripts/python.exe', 'venv/Scripts/python.exe', '.venv/bin/python', 'venv/bin/python'):
                add_python(folder / relative)
            if depth < 6:
                pending.extend((p, depth + 1) for p in children
                               if p.name.lower() not in SKIP | {'.venv', 'venv'})
        except (OSError, ValueError):
            continue
    if limited:
        notes.append('扫描达到时间/目录上限；可点击“扫描指定目录”继续查找。')
    # 优先检测用户明确指定的解释器和仓库旁的环境，避免耗费时间探测不相关环境。
    preferred = []
    for value in [current.get('vggt_python'), env.get('VGGT_PYTHON')]:
        if value and Path(value).is_file():
            preferred.append(canonical(value))
    for repo in result['vggt_repo']:
        repo = Path(repo)
        # 常见 VGGT 布局是根目录/.venv 与根目录/src/demo_colmap.py，
        # 所以既查仓库内，也查 src 的父目录，避免错过现成的项目环境。
        for base in (repo, repo.parent):
            for relative in ('.venv/Scripts/python.exe', 'venv/Scripts/python.exe',
                             '.venv/bin/python', 'venv/bin/python'):
                path = base / relative
                if path.is_file():
                    preferred.append(canonical(path))
    python_paths = list(dict.fromkeys(preferred + python_paths))
    checks = {}
    for path in python_paths[:probe_limit]:
        check()
        report({'type': 'discovery_status', 'text': '检查 Python 核心依赖…'})
        ok, message = python_probe(path, check)
        checks[path] = message
        if ok:
            add('vggt_python', path)
    if len(python_paths) > probe_limit:
        notes.append('Python 候选较多，仅检查前十个；可以选择解释器或缩小扫描目录。')
    return {'candidates': result, 'python_checks': checks, 'notes': notes,
            'directories_scanned': len(visited)}
