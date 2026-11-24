"""Windows 双击启动入口：使用快捷方式指定的虚拟环境，不弹出命令行窗口。

启动异常写入 logs 并显示错误对话框，避免 pythonw 启动失败时没有提示。
执行 python launch.py --check 只检查导入，不启动窗口。
"""
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback


def main():
    project = Path(__file__).resolve().parent
    os.chdir(project)
    if '--check' in sys.argv:
        from glacier.ui import run  # noqa: F401
        print('Launcher dependency check passed.')
        return 0

    log_path = project / 'logs' / f'launch_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log'
    try:
        log_path.parent.mkdir(exist_ok=True)
        log = log_path.open('w', encoding='utf-8', buffering=1)
    except OSError as exc:
        show_error(f'无法创建启动日志：{exc}')
        return 1

    # pythonw 没有控制台；显式保留输出，便于诊断依赖和渲染问题。
    sys.stdout = sys.stderr = log

    def report_error(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb, file=log)
        log.flush()
        show_error(f'程序发生错误：{exc_value}\n\n详细日志：\n{log_path}')

    sys.excepthook = report_error
    try:
        from glacier.ui import run
        return run()
    except Exception:
        report_error(*sys.exc_info())
        return 1


def show_error(message):
    if os.name == 'nt':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, '冰川三维重建 · 启动提示', 0x10)
    else:
        print(message, file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
