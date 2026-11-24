"""工作线程只发送数据，不访问 Qt 控件。取消使用线程安全 Event。"""
import threading
from PySide6.QtCore import QThread, Signal
from .pipeline import Cancelled


class Worker(QThread):
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def check_cancel(self):
        if self.cancelled.is_set():
            raise Cancelled('任务已取消；已完成的文件仍保留。')

    def run(self):
        try:
            self.result.emit(self.fn(self.progress.emit, self.check_cancel))
        except Exception as exc:
            self.error.emit(str(exc))
