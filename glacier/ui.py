"""Qt 主界面。后台线程只传递数据，所有控件更新都在主线程执行。"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import shutil

import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QPoint, QRect, QRectF
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QRegion,
                            QPen, QRadialGradient, QPixmap)
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog,
    QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QInputDialog, QSlider,
    QSpinBox, QStyle, QStyleOptionGroupBox, QTabBar, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)

from .evaluation import APIConfig, evaluate, validate_api
from .pipeline import PipelineConfig
from .discovery import KEYS, discover, is_tool_path
from .preprocessing import PreprocessConfig, collect_images, preprocess, read_rgb, resize_long_edge
from .tasks import reconstruction_task, save_json
from .viewer import ModelViewer
from .workers import Worker


class DropList(QListWidget):
    """专用拖放区，也允许文件夹；文件列表的扫描在后台执行。"""
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setMinimumHeight(140)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.files_dropped.emit([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        event.acceptProposedAction()

    def paths(self):
        return [Path(self.item(i).text()) for i in range(self.count())]

    def paintEvent(self, event):
        # Keep the empty photo-drop area transparent so it blends into the
        # surrounding landscape instead of becoming a large dark slab.
        super().paintEvent(event)


class GlassPillButton(QPushButton):
    """A photo-action button painted as a clean, truly elliptical glass slot."""
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(.75, .75, -.75, -.75)
        if not self.isEnabled():
            fill, edge, foreground = QColor(190, 218, 242, 15), QColor(230, 242, 255, 55), QColor(220, 232, 244, 125)
        elif self.isDown():
            fill, edge, foreground = QColor(130, 175, 220, 95), QColor(242, 249, 255, 205), QColor(255, 255, 255)
        elif self.underMouse():
            fill, edge, foreground = QColor(190, 218, 242, 72), QColor(242, 249, 255, 205), QColor(255, 255, 255)
        else:
            fill, edge, foreground = QColor(190, 218, 242, 34), QColor(230, 242, 255, 125), QColor(245, 250, 255)
        radius = rect.height() / 2
        painter.setPen(QPen(edge, 1.5))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, radius, radius)
        painter.setPen(foreground)
        painter.setFont(self.font())
        painter.drawText(rect.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class AtmosphereWidget(QWidget):
    """Procedural animated navy backdrop; no remote asset is required."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0
        self._frame = QImage()
        self._frost = QImage()
        self._frame_size = None
        # Bundled low-resolution snow landscape from the user-supplied visual
        # reference. Keep a dark procedural fallback for moved/partial installs.
        self.wallpaper = QImage(str(Path(__file__).resolve().parent.parent / 'assets' / 'snow-peaks-wallpaper.jpg'))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.advance)
        self.clock.start()

    def advance(self):
        self.phase = (self.phase + 0.0035) % (2 * math.pi)
        self._frame_size = None
        self.update()
        for surface in self.findChildren(QWidget):
            if surface.property("frostSurface"):
                surface.update()

    def ensure_frame(self):
        """Cache the scene and a 12 logical-pixel Gaussian-blurred backdrop.

        Only the atmosphere is sampled: controls and text are never blurred.
        Rendering at half resolution bounds the animated painting cost.
        """
        size = (max(1, self.width()), max(1, self.height()))
        if self._frame_size == size:
            return
        from PIL import Image, ImageFilter
        frame = QImage(max(1, size[0] // 2), max(1, size[1] // 2),
                       QImage.Format.Format_RGBA8888)
        frame.fill(QColor('#05080e'))
        painter = QPainter(frame)
        self.draw_scene(painter, frame.width(), frame.height())
        painter.end()
        pixels = Image.frombytes('RGBA', (frame.width(), frame.height()),
                                 bytes(frame.constBits()), 'raw', 'RGBA', frame.bytesPerLine())
        blurred = pixels.filter(ImageFilter.GaussianBlur(6))
        self._frame = frame
        self._frost = QImage(blurred.tobytes(), frame.width(), frame.height(),
                             QImage.Format.Format_RGBA8888).copy()
        self._frame_size = size

    def paintEvent(self, event):
        self.ensure_frame()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.rect(), self._frame)

    def draw_scene(self, painter, width, height):
        """Draw the reference wallpaper and subdued navy veil."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        bounds = QRectF(0, 0, width, height)
        painter.fillRect(bounds, QColor('#05080e'))
        if not self.wallpaper.isNull():
            # Cover the available area without stretching the landscape.
            source = QRectF(self.wallpaper.rect())
            source_aspect = source.width() / max(1, source.height())
            target_aspect = width / max(1, height)
            if source_aspect > target_aspect:
                crop_width = source.height() * target_aspect
                source.setX((source.width() - crop_width) / 2)
                source.setWidth(crop_width)
            else:
                crop_height = source.width() / target_aspect
                source.setY((source.height() - crop_height) / 2)
                source.setHeight(crop_height)
            painter.drawImage(bounds, self.wallpaper, source)
        else:
            base = QLinearGradient(0, 0, width, height)
            base.setColorAt(0.0, QColor('#101B2A'))
            base.setColorAt(0.46, QColor('#0B1725'))
            base.setColorAt(1.0, QColor('#060B13'))
            painter.fillRect(bounds, base)

        # The reference keeps photography visible while a navy veil protects
        # foreground contrast; darken edges and the header without flattening it.
        painter.fillRect(bounds, QColor(4, 7, 13, 72))
        top_veil = QLinearGradient(0, 0, 0, height * .24)
        top_veil.setColorAt(0, QColor(4, 7, 13, 150))
        top_veil.setColorAt(1, QColor(4, 7, 13, 0))
        painter.fillRect(bounds, top_veil)

        # 多个大而虚的灰蓝色雾团彼此叠加，形成参考截图中的云雾纹理；
        # 透明度保持克制，不形成亮斑。频率为整数，动画循环无跳变。
        t = self.phase
        clouds = [
            (.12, .18, .19, .25, 5, 0.0),
            (.34, .34, .21, .18, 4, 1.2),
            (.72, .18, .18, .23, 5, 2.4),
            (.91, .48, .22, .26, 5, 3.1),
            (.63, .76, .23, .21, 4, 5.1),
            (.13, .84, .2, .22, 4, 2.8),
        ]
        for index, (x, y, rx, ry, alpha, offset) in enumerate(clouds):
            drift_x = .018 * math.sin(t + offset)
            drift_y = .022 * math.cos(t + offset)
            cx = width * (x + drift_x)
            cy = height * (y + drift_y)
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(-18 + 9 * math.sin(t + offset))
            painter.scale(width * rx, height * ry)
            mist = QRadialGradient(QPointF(0, 0), 1.0)
            mist.setColorAt(0, QColor(105, 132, 158, alpha))
            mist.setColorAt(.34, QColor(82, 108, 136, int(alpha * .68)))
            mist.setColorAt(.72, QColor(57, 80, 106, int(alpha * .24)))
            mist.setColorAt(1, QColor(39, 57, 77, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(mist)
            painter.drawEllipse(QPointF(0, 0), 1.0, 1.0)
            painter.restore()

        # Match the reference's dark veil: the center remains legible while
        # the outer edge and top are gently shaded behind the glass panels.
        vignette = QRadialGradient(QPointF(width * .5, height * .46),
                                   max(width, height) * .78)
        vignette.setColorAt(0, QColor(4, 7, 13, 0))
        vignette.setColorAt(.30, QColor(4, 7, 13, 0))
        vignette.setColorAt(.72, QColor(4, 7, 13, 78))
        vignette.setColorAt(1, QColor(4, 7, 13, 170))
        painter.fillRect(bounds, vignette)


def paint_frost(widget, painter, rect, radius=18, top_gap=None):
    """Paint a translucent frosted panel and a clean hairline rim."""
    root = widget.window().centralWidget()
    if not isinstance(root, AtmosphereWidget):
        return
    root.ensure_frame()
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    painter.setClipPath(path)
    position = widget.mapTo(root, QPoint(0, 0))
    sx = root._frost.width() / max(1, root.width())
    sy = root._frost.height() / max(1, root.height())
    source = QRectF((position.x() + rect.x()) * sx,
                    (position.y() + rect.y()) * sy,
                    rect.width() * sx, rect.height() * sy)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.drawImage(rect, root._frost, source)
    # A light navy tint preserves legibility while allowing the glacier
    # wallpaper to remain visible through the glass, as in the reference.
    painter.fillRect(rect, QColor(25, 35, 50, 46))
    sheen = QLinearGradient(rect.topLeft(), rect.bottomRight())
    sheen.setColorAt(0, QColor(208, 231, 249, 8))
    sheen.setColorAt(.35, QColor(208, 231, 249, 0))
    sheen.setColorAt(1, QColor(188, 219, 244, 4))
    painter.fillRect(rect, sheen)
    painter.setClipping(False)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # Leave the title's span clear so the frame line does not run through it.
    if top_gap:
        gap_region = QRegion(widget.rect()).subtracted(
            QRegion(QRect(top_gap[0], round(rect.top() - 3), top_gap[1], 7)))
        painter.save()
        painter.setClipRegion(gap_region)
    painter.setPen(QPen(QColor(255, 255, 255, 33), 1))
    painter.drawRoundedRect(rect.adjusted(.5, .5, -.5, -.5), radius, radius)
    if top_gap:
        painter.restore()
    painter.restore()


def paint_title_slot(widget, painter, rect):
    """Give a group-box heading the light, recessed glass-pill treatment."""
    root = widget.window().centralWidget()
    if not isinstance(root, AtmosphereWidget):
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
    painter.setClipPath(path)

    painter.fillRect(rect, QColor(17, 24, 38, 255))

    # Keep a subtle gloss while the solid base blocks background details.
    gloss = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    gloss.setColorAt(0, QColor(242, 249, 255, 52))
    gloss.setColorAt(.48, QColor(212, 232, 249, 16))
    gloss.setColorAt(1, QColor(143, 178, 210, 12))
    painter.fillRect(rect, gloss)

    painter.setClipping(False)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(235, 246, 255, 88), 1))
    painter.drawRoundedRect(rect.adjusted(.5, .5, -.5, -.5),
                            rect.height() / 2, rect.height() / 2)
    painter.setPen(QPen(QColor(255, 255, 255, 48), 1))
    painter.drawLine(QPointF(rect.left() + 16, rect.top() + 2),
                     QPointF(rect.right() - 16, rect.top() + 2))
    painter.restore()


class FrostGroupBox(QGroupBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProperty('frostSurface', True)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Use Qt's actual title geometry, then size the heading with the same
        # installed bold Chinese font used by the stylesheet.
        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        title_font = QFont('Noto Sans SC')
        title_font.setPixelSize(18)
        title_font.setWeight(QFont.Weight.Black)
        option.font = title_font
        option.text = self.title()
        title_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_GroupBox, option,
            QStyle.SubControl.SC_GroupBoxLabel, self)
        # Measure the same string with the same font; use this exact extent for
        # both the frame notch and blue marker placement.
        text_width = QFontMetrics(title_font).horizontalAdvance(self.title())
        # Reserve a fixed slot for the accent so QSS padding and the platform
        # style geometry cannot shift the marker into the visible title.
        text_left = 32
        marker_width = 6
        marker_gap = 6
        marker_x = text_left - marker_gap - marker_width
        gap_start = max(8, marker_x - 5)
        gap_end = text_left + text_width + 8
        paint_frost(self, painter, QRectF(self.rect()).adjusted(0, 12, 0, 0),
                    top_gap=(gap_start, gap_end - gap_start))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#168FEA'))
        # Put the blue accent immediately to the left of the heading and
        # align its vertical center with the heading's actual label center.
        marker_y = title_rect.center().y() - 6
        painter.drawRoundedRect(QRectF(marker_x, marker_y, marker_width, 18), 1.5, 1.5)
        painter.restore()
        painter.end()
        super().paintEvent(event)


class FrostLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProperty('frostSurface', True)

    def paintEvent(self, event):
        painter = QPainter(self)
        paint_frost(self, painter, QRectF(self.rect()), radius=14)
        painter.end()
        super().paintEvent(event)


class GlassTabBar(QTabBar):
    """Paint each page tab as a rounded translucent glass slot."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setExpanding(False)
        self.setUsesScrollButtons(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = self.font()
        font.setPixelSize(16)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        for index in range(self.count()):
            rect = QRectF(self.tabRect(index)).adjusted(2, 2, -2, -2)
            selected = index == self.currentIndex()
            if selected:
                fill = QColor(22, 143, 234, 100)
                edge = QColor(102, 190, 255, 235)
                text = QColor(255, 255, 255, 255)
            elif self.tabAt(self.mapFromGlobal(self.cursor().pos())) == index and self.underMouse():
                fill = QColor(190, 218, 242, 58)
                edge = QColor(238, 247, 255, 175)
                text = QColor(235, 244, 252, 240)
            else:
                fill = QColor(190, 218, 242, 28)
                edge = QColor(230, 242, 255, 105)
                text = QColor(235, 244, 252, 220)
            painter.setPen(QPen(edge, 1))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
            painter.setPen(text)
            painter.drawText(rect.adjusted(8, 0, -8, 0),
                             Qt.AlignmentFlag.AlignCenter, self.tabText(index))
        painter.end()


class FrostTabs(QTabWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTabBar(GlassTabBar())
        self.setProperty('frostSurface', True)

    def paintEvent(self, event):
        painter = QPainter(self)
        paint_frost(self, painter, QRectF(self.rect()).adjusted(0, self.tabBar().height(), 0, 0))
        painter.end()
        super().paintEvent(event)


class EvaluationPanel(QWidget):
    """评分不假设满分或量纲，指标和值直接取自 API，并保留原始 JSON。"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.score = QLabel('尚未评价')
        self.score.setObjectName('sectionTitle')
        layout.addWidget(self.score)
        self.metrics = QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(['后端指标', '后端返回值'])
        self.metrics.horizontalHeader().setStretchLastSection(True)
        self.metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.metrics)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText('API 评价文本')
        layout.addWidget(self.text)
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setPlaceholderText('原始 JSON 响应')
        layout.addWidget(self.raw)

    def reset(self, message='尚未评价'):
        self.score.setText(message)
        self.metrics.setRowCount(0)
        self.text.clear()
        self.raw.clear()

    def display(self, payload):
        result = payload['result']
        # 没有 score 字段时如实标注，不计算或猜测一个评分。
        score = result.get('score', '后端未提供 score')
        self.score.setText('评分：' + str(score) + (' / 100' if isinstance(score, (int, float)) else ''))
        self.score.setTextFormat(Qt.TextFormat.PlainText)
        metrics = result.get('metrics', {})
        if isinstance(metrics, list):
            self.metrics.setRowCount(len(metrics))
            for row, item in enumerate(metrics):
                self.metrics.setItem(row, 0, QTableWidgetItem(str(item.get('name', '未命名指标'))))
                value = f"{item.get('score', '?')} / {item.get('max_score', '?')} · {item.get('evidence', '')}"
                self.metrics.setItem(row, 1, QTableWidgetItem(value))
        else:
            if not isinstance(metrics, dict):
                metrics = {'metrics': metrics}
            self.metrics.setRowCount(len(metrics))
            for row, (key, value) in enumerate(metrics.items()):
                self.metrics.setItem(row, 0, QTableWidgetItem(str(key)))
                self.metrics.setItem(row, 1, QTableWidgetItem(
                    json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)))
        text = result.get('text', result.get('evaluation', result.get('summary', result.get('message', ''))))
        self.text.setPlainText(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2))
        self.raw.setPlainText(json.dumps(payload['raw'], ensure_ascii=False, indent=2))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('冰川三维重建 · Glacier Reconstruction Studio')
        icon_path = Path(__file__).resolve().parent.parent / 'assets' / 'glacier_icon.ico'
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        # Keep the workstation compact; each results column has its own
        # scrollbar for the model details and API evaluation below the preview.
        self.resize(1100, 760)
        self.setMinimumSize(820, 620)
        self.setAcceptDrops(True)
        self.workers = set()       # 保持强引用，防止运行中的 QThread 被析构
        self.job = None
        self.preview_worker = None
        self.preview_revision = 0
        self.models = {}
        self.evaluations = {}
        self.last_run = None
        self.pending_cleanup_run = None
        self.closing = False
        self._active_vggt_dialog = None
        self._build()
        # 窗口先显示，再在工作线程恢复/发现工具，避免遍历安装目录卡住 GUI。
        QTimer.singleShot(300, self.auto_detect_tools)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)   # 调参防抖，避免每次按键都启动计算
        self.preview_timer.timeout.connect(self.launch_preview)
        self.detected_tools = {}
        self.restore_tool_paths()
        for field in self.tool_path_fields().values():
            field.editingFinished.connect(self.cache_tool_paths)
        self.restore_session()

    def session_file(self):
        return Path(__file__).resolve().parents[1] / '.local' / 'session.json'

    def save_session(self):
        """Persist resumable UI state without API credentials or machine secrets."""
        try:
            save_json(self.session_file(), {
                'version': 1,
                'output': self.output.text().strip(),
                'images': [str(path) for path in self.drop.paths()],
                'last_run': self.last_run,
                'models': self.models,
                'evaluations': self.evaluations,
                'pending_cleanup_run': self.pending_cleanup_run,
            })
        except (OSError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(f'无法保存恢复状态：{exc}')

    def restore_session(self):
        """Restore the last photo list and task results when the app reopens."""
        session_path = self.session_file()
        try:
            state = json.loads(session_path.read_text(encoding='utf-8')) if session_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            state = {}

        saved_output = state.get('output')
        if saved_output:
            self.output.setText(str(saved_output))
        self.last_run = state.get('last_run')
        self.pending_cleanup_run = state.get('pending_cleanup_run')
        if not self.last_run:
            try:
                manifests = Path(self.output.text()).expanduser().glob('run_*/manifest.json')
                latest = max(manifests, key=lambda path: path.stat().st_mtime, default=None)
                if latest:
                    self.last_run = str(latest.parent)
            except OSError:
                pass

        # Existing sessions restore original inputs when available. If those
        # files moved, use the processed copies retained in the task folder.
        has_saved_image_list = isinstance(state.get('images'), list)
        image_paths = state.get('images', []) if has_saved_image_list else []
        run_path = Path(self.last_run) if self.last_run else None
        manifest_path = run_path / 'manifest.json' if run_path else None
        manifest = {}
        if manifest_path and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        if not has_saved_image_list and manifest:
            for item in manifest.get('input_images', []):
                source = Path(item.get('source', ''))
                processed = run_path / 'images' / item.get('processed', '')
                image_paths.append(str(source if source.is_file() else processed))
        restored = []
        for raw_path in image_paths:
            path = Path(raw_path)
            if path.is_file():
                self.drop.addItem(str(path))
                restored.append(path)
        if manifest:
            restored_paths = {str(path.resolve()).casefold() for path in restored}
            for item in manifest.get('input_images', []):
                source = Path(item.get('source', ''))
                processed = run_path / 'images' / item.get('processed', '')
                processed_key = str(processed.resolve()).casefold()
                source_was_saved = str(source) in image_paths
                if (not source.is_file() and processed.is_file() and processed_key not in restored_paths
                        and (not has_saved_image_list or source_was_saved)):
                    self.drop.addItem(str(processed))
                    restored.append(processed)
                    restored_paths.add(processed_key)
        self.count_label.setText(f'{len(restored)} 张')
        if restored:
            self.drop.setCurrentRow(0)
            self.queue_preview()

        if run_path and run_path.is_dir():
            self.run_path.setText('任务输出：' + str(run_path))
        model_paths = dict(state.get('models', {}))
        saved_evaluations = state.get('evaluations', {})
        if manifest:
            for method, result in manifest.get('results', {}).items():
                model_paths.setdefault(method, result.get('model'))
        for method in ('A', 'B'):
            raw_path = model_paths.get(method)
            if raw_path and Path(raw_path).is_file():
                self.models[method] = str(raw_path)
                try:
                    self.viewers[method].load_model(raw_path)
                except Exception as exc:
                    self.log.appendPlainText(f'恢复路径 {method} 模型预览失败：{exc} · {raw_path}')
                if isinstance(saved_evaluations.get(method), dict):
                    data = saved_evaluations[method]
                    self.evaluations[method] = data
                    self.eval_panels[method].display(data)
                else:
                    evaluation_path = (run_path / method / 'evaluation.json') if run_path else None
                    if evaluation_path and evaluation_path.is_file():
                        try:
                            data = json.loads(evaluation_path.read_text(encoding='utf-8'))
                            self.evaluations[method] = data
                            self.eval_panels[method].display(data)
                        except (OSError, json.JSONDecodeError, KeyError, TypeError):
                            pass
        if self.models:
            self.tabs.setCurrentIndex(1)
        if self.last_run or restored or self.models:
            self.statusBar().showMessage('已恢复上次照片列表和任务结果；新建任务后将更新恢复记录。')

    def _build(self):
        root = AtmosphereWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 8, 22, 8)
        layout.setSpacing(6)
        brand = QLabel("Let's START")
        brand.setObjectName('eyebrow')
        layout.addWidget(brand)
        self.title_label = QLabel('冰川三维重建工作台')
        self.title_label.setObjectName('appTitle')
        layout.addWidget(self.title_label)
        header_rule = QWidget()
        header_rule.setFixedHeight(1)
        header_rule.setStyleSheet('background: rgba(255,255,255,34);')
        layout.addWidget(header_rule)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        layout.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Keep the control rail compact and prevent a horizontal scrollbar from
        # appearing when long filesystem paths are present in the editors.
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(4, 2, 4, 2)
        side.setSpacing(6)
        scroll.setWidget(left)
        splitter.addWidget(scroll)
        self._preprocess_panel(side)
        self._algorithm_panel(side)
        self._api_panel(side)
        side.addStretch()
        self.tabs = FrostTabs()
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 760])
        self._input_page()
        self._results_page()
        self._logs_page()
        # Very light cool rim on the outer glass cards. Child widgets stay
        # transparent so the procedural atmosphere remains visible beneath.
        self.tabs.setObjectName('glassTabs')
        scroll.setObjectName('glassRail')
        for panel in (self.tabs, scroll):
            panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        footer = QHBoxLayout()
        self.run_button = QPushButton('开始重建')
        self.run_button.setObjectName('primaryButton')
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton('取消任务')
        self.cancel_button.setObjectName('primaryButton')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.stage_label = QLabel('就绪')
        self.stage_label.setMinimumWidth(104)
        self.stage_label.setObjectName('statusText')
        footer.addWidget(self.run_button)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.progress, 1)
        footer.addWidget(self.stage_label)
        layout.addLayout(footer)
        self.statusBar().showMessage('可拖入单张/多张照片或文件夹；重建需要至少两张重叠影像。')

    def _preprocess_panel(self, layout):
        box = FrostGroupBox('冰雪影像预处理')
        form = QFormLayout(box)
        form.setContentsMargins(8, 12, 8, 6)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(8)
        self.prep_widgets = {}
        specs = [
            ('highlight', '雪面高光压缩', 0, .8, .05, .15, '压缩高亮区域；不能恢复饱和像素。'),
            ('clahe', '局部对比度 CLAHE', 0, 6, .1, 1.5, '适度增强弱纹理；0 关闭，过强会放大噪声。'),
            ('gamma', 'Gamma 亮度', .4, 2.5, .05, 1, '<1 提亮；>1 压暗。'),
            ('denoise', '双边降噪强度', 0, 60, 2, 0, '默认关闭，以免抹去冰雪弱纹理。'),
            ('sharpen', '轻度锐化', 0, 1, .05, 0, '默认关闭；过度锐化可能引入伪边缘。'),
        ]
        for name, label, low, high, step, default, tip in specs:
            widget = QDoubleSpinBox()
            widget.setRange(low, high)
            widget.setSingleStep(step)
            widget.setValue(default)
            widget.valueChanged.connect(self.queue_preview)
            self.prep_widgets[name] = widget
            form.addRow(label, widget)
        edge = QSpinBox()
        edge.setRange(0, 12000)
        edge.setSingleStep(200)
        edge.setValue(2400)
        edge.setSpecialValueText('保留原尺寸')
        edge.valueChanged.connect(self.queue_preview)
        self.prep_widgets['max_edge'] = edge
        form.addRow('输出最长边 / px', edge)
        reset = QPushButton('恢复保守默认参数')
        reset.clicked.connect(self.reset_preprocessing)
        form.addRow(reset)
        layout.addWidget(box)

    def _path_field(self, form, label, mode, initial=''):
        field = QLineEdit(initial)
        field.setObjectName('pathField')
        field.setMinimumWidth(0)
        # 路径通常会被输入框宽度截断；悬停时展示当前完整值，且
        # 随用户输入、浏览选择或配置加载而同步更新。
        field.setToolTip(field.text())
        field.textChanged.connect(field.setToolTip)
        button = QPushButton('浏览')
        button.setObjectName('browseButton')
        button.setFixedWidth(44)
        button.setFixedHeight(30)
        def choose():
            if mode == 'dir':
                value = QFileDialog.getExistingDirectory(self, label, field.text())
            else:
                value, _ = QFileDialog.getOpenFileName(self, label, field.text())
            if value:
                field.setText(value)
        button.clicked.connect(choose)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(field, 1)
        row.addWidget(button)
        form.addRow(label, row)
        return field

    def _algorithm_panel(self, layout):
        box = FrostGroupBox('重建路径与输出')
        form = QFormLayout(box)
        form.setContentsMargins(8, 12, 8, 6)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(6)
        # Long Windows paths deserve the full rail width: put field labels above inputs.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.path_a = QCheckBox('路径 A · VGGT')
        self.path_a.setChecked(True)
        self.path_b = QCheckBox('路径 B · COLMAP + OpenMVS')
        # The full label is useful, but at compact window widths its natural
        # one-line size can force the entire scroll area wider than the rail.
        form.addRow(self.path_a)
        form.addRow(self.path_b)
        self.output = self._path_field(form, '结果保存目录', 'dir', str(Path(__file__).resolve().parents[1] / 'outputs'))
        self.vggt_python = self._path_field(form, 'VGGT Python', 'file')
        self.vggt_repo = self._path_field(form, 'VGGT 仓库目录', 'dir')
        self.vggt_args = QLineEdit('[]')
        self.vggt_args.setPlaceholderText('["--use_ba"]')
        form.addRow('VGGT 参数 JSON', self.vggt_args)
        self.colmap = self._path_field(form, 'COLMAP 程序', 'file', 'colmap')
        self.openmvs = self._path_field(form, 'OpenMVS bin', 'dir')
        self.colmap_extra = QLineEdit('{}')
        self.openmvs_extra = QLineEdit('{}')
        for label, field in [('COLMAP 阶段参数', self.colmap_extra), ('OpenMVS 阶段参数', self.openmvs_extra)]:
            form.addRow(label, field)
        row = QHBoxLayout()
        save = QPushButton('保存配置')
        save.clicked.connect(self.save_settings)
        load = QPushButton('加载配置')
        load.clicked.connect(self.load_settings)
        row.addWidget(save)
        row.addWidget(load)
        form.addRow(row)
        detect_row = QHBoxLayout()
        self.detect_button = QPushButton('自动检测工具')
        self.detect_button.clicked.connect(self.auto_detect_tools)
        scan = QPushButton('扫描指定目录')
        scan.clicked.connect(self.choose_scan_roots)
        detect_row.addWidget(self.detect_button)
        detect_row.addWidget(scan)
        form.addRow(detect_row)
        self.candidates_button = QPushButton('选择工具版本')
        self.candidates_button.setEnabled(False)
        self.candidates_button.clicked.connect(self.choose_detected_tools)
        form.addRow(self.candidates_button)
        self.discovery_label = QLabel('启动后自动查找已安装工具')
        self.discovery_label.setWordWrap(True)
        self.discovery_label.setTextFormat(Qt.TextFormat.PlainText)
        # The scroll rail must grow to the complete wrapped message. QLabel's
        # height-for-width hint is not always honored inside QFormLayout, so we
        # explicitly size it after layout and whenever the window is resized.
        form.addRow(self.discovery_label)
        layout.addWidget(box)

    def _api_panel(self, layout):
        box = FrostGroupBox('模型质量评价 API')
        form = QFormLayout(box)
        form.setContentsMargins(8, 12, 8, 6)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(6)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.api_enabled = QCheckBox('每条路径完成后自动上传并评价')
        form.addRow(self.api_enabled)
        self.api_url = QLineEdit()
        self.api_url.setPlaceholderText('https://api.deepseek.com 或自建评价接口完整地址')
        self.api_url.setToolTip(self.api_url.text())
        self.api_url.textChanged.connect(self.api_url.setToolTip)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.auth_header = QLineEdit('Authorization')
        self.auth_prefix = QLineEdit('Bearer')
        self.file_field = QLineEdit('model')
        self.meta_field = QLineEdit('metadata')
        self.result_path = QLineEdit()
        self.result_path.setPlaceholderText('留空，或 data.result')
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 1800)
        self.timeout.setValue(120)
        self.timeout.setSuffix(' 秒')
        for label, field in [('API 完整地址', self.api_url), ('密钥（仅内存）', self.api_key),
                             ('鉴权头名称', self.auth_header), ('鉴权前缀', self.auth_prefix),
                             ('文件字段', self.file_field), ('元数据字段', self.meta_field),
                             ('响应结果根路径', self.result_path), ('响应读取超时', self.timeout)]:
            form.addRow(label, field)
        layout.addWidget(box)

    def _input_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel('将单张 / 多张照片拖到此处，也可以直接选择图片文件夹')
        label.setObjectName('sectionTitle')
        layout.addWidget(label)
        row = QHBoxLayout()
        for text, callback in [('选择图片', self.choose_files), ('选择文件夹', self.choose_folder),
                               ('移除选中', self.remove_selected), ('清空列表', self.clear_images)]:
            button = GlassPillButton(text)
            button.setObjectName('imageActionButton')
            button.clicked.connect(callback)
            row.addWidget(button)
        self.count_label = QLabel('0 张')
        row.addWidget(self.count_label)
        row.addStretch()
        layout.addLayout(row)
        self.drop = DropList()
        self.drop.setObjectName('glassDropList')
        self.drop.setMinimumHeight(96)
        self.drop.files_dropped.connect(self.import_paths)
        self.drop.currentRowChanged.connect(self.queue_preview)
        layout.addWidget(self.drop, 1)
        self.preview_note = QLabel('选择一张图像，调整左侧参数即可更新预览')
        self.preview_note.setWordWrap(True)
        self.preview_note.hide()
        pair = QHBoxLayout()
        self.original_preview = self._preview_label('原图')
        self.processed_preview = self._preview_label('预处理预览')
        pair.addWidget(self.original_preview, 1)
        pair.addWidget(self.processed_preview, 1)
        layout.addLayout(pair, 3)
        self.tabs.addTab(page, '① 图像与预处理')

    def _preview_label(self, name):
        label = FrostLabel(name)
        label.setObjectName('previewLabel')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(160, 140)
        label.setStyleSheet('background:transparent;border:0;color:#D5E4F1;')
        return label

    def _results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        # Each algorithm gets its own vertical scroll area so its model canvas,
        # footer text and API evaluation move together independently.
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.viewers, self.eval_panels = {}, {}
        for method, name in [('A', '路径 A · VGGT'), ('B', '路径 B · COLMAP + OpenMVS')]:
            content = QWidget()
            content.setMinimumHeight(800)
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(2, 2, 2, 8)
            viewer = ModelViewer(name, vggt_controls=(method == 'A'))
            viewer.load_requested.connect(lambda m=method: self.load_model(m))
            viewer.evaluate_requested.connect(lambda m=method: self.evaluate_model(m))
            self.viewers[method] = viewer
            evaluation = EvaluationPanel()
            self.eval_panels[method] = evaluation
            # Evaluation details can be long; scroll them inside their own
            # pane so they never take space from the model preview or footer.
            evaluation_scroll = QScrollArea()
            evaluation_scroll.setWidgetResizable(True)
            evaluation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            evaluation_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            evaluation_scroll.setWidget(evaluation)
            content_layout.addWidget(viewer, 0)
            content_layout.addWidget(evaluation_scroll, 1)
            side_scroll = QScrollArea()
            side_scroll.setObjectName('resultsScroll')
            side_scroll.setWidgetResizable(True)
            side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            # Keep each scrollbar available and independently draggable.
            side_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            viewer.scroll_area = side_scroll
            side_scroll.setWidget(content)
            split.addWidget(side_scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([1, 1])
        layout.addWidget(split, 1)
        self.tabs.addTab(page, '② 模型与评价对比')

    def _logs_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.run_path = QLabel('任务输出：尚未运行')
        self.run_path.setWordWrap(True)
        self.run_path.setTextFormat(Qt.TextFormat.PlainText)
        self.run_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.run_path)
        note = QLabel('进度按已完成阶段估算；单个 GPU 步骤可能长时间停留。完整日志及 manifest.json 保存在独立任务目录。')
        note.hide()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log)
        self.tabs.addTab(page, '③ 任务日志')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.import_paths([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        event.acceptProposedAction()

    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, '选择冰川影像', '',
            '图像 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)')
        if files:
            self.import_paths(files)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择图像文件夹（递归导入）')
        if folder:
            self.import_paths([folder])

    def import_paths(self, paths):
        if not paths or self.job:
            if self.job:
                self.statusBar().showMessage('请等待当前任务完成，再导入新图像。')
            return
        def scan(report, check):
            check()
            found = collect_images(paths)
            check()
            return found
        self.start_worker(scan, self.import_finished, primary=True)
        self.stage_label.setText('扫描图像文件…')

    def import_finished(self, found):
        existing = set(self.drop.paths())
        for path in found:
            if path not in existing:
                self.drop.addItem(str(path))
                existing.add(path)
        self.count_label.setText(f'{self.drop.count()} 张')
        if self.drop.currentRow() < 0 and self.drop.count():
            self.drop.setCurrentRow(0)
        self.save_session()
        self.statusBar().showMessage(f'已加载 {self.drop.count()} 张图像；损坏文件会在预览或任务中报告。')

    def remove_selected(self):
        if self.job:
            return
        for item in self.drop.selectedItems():
            self.drop.takeItem(self.drop.row(item))
        self.count_label.setText(f'{self.drop.count()} 张')
        self.queue_preview()
        self.save_session()

    def clear_images(self):
        if self.job:
            return
        if self.last_run and not self.pending_cleanup_run:
            self.pending_cleanup_run = self.last_run
        self.drop.clear()
        self.count_label.setText('0 张')
        self.original_preview.setText('原图')
        self.processed_preview.setText('预处理预览')
        self.queue_preview()
        self.save_session()
        if self.pending_cleanup_run:
            self.statusBar().showMessage('已清空导入照片；下次重建成功后将清理上一个任务目录。')

    def prep_config(self):
        return PreprocessConfig(**{k: w.value() for k, w in self.prep_widgets.items()})

    def tool_config(self):
        return PipelineConfig(self.vggt_python.text().strip(), self.vggt_repo.text().strip(),
            self.vggt_args.text(), self.colmap.text().strip(), self.openmvs.text().strip(),
            self.colmap_extra.text(), self.openmvs_extra.text())

    def api_config(self):
        return APIConfig(self.api_enabled.isChecked(), self.api_url.text().strip(), self.api_key.text(),
            self.auth_header.text().strip(), self.auth_prefix.text(), self.file_field.text().strip(),
            self.meta_field.text().strip(), self.result_path.text().strip(), self.timeout.value())

    def reset_preprocessing(self):
        for key, value in asdict(PreprocessConfig()).items():
            self.prep_widgets[key].setValue(value)

    def queue_preview(self, *_):
        self.preview_revision += 1
        if hasattr(self, 'preview_timer'):
            self.preview_timer.start()

    def launch_preview(self):
        item = self.drop.currentItem()
        if not item or self.preview_worker or self.closing:
            return
        revision, path, cfg = self.preview_revision, item.text(), self.prep_config()
        def compute(report, check):
            rgb = read_rgb(path)
            check()
            saturated = float((rgb.max(axis=2) >= 250).mean() * 100)
            original = resize_long_edge(rgb, 1100)
            processed = preprocess(original, replace(cfg, max_edge=min(cfg.max_edge or 1100, 1100)))
            check()
            return (revision, original, processed, saturated, path)
        worker = self.start_worker(compute, self.show_preview, primary=False,
                                   error=lambda msg: self.preview_note.setText('预览失败：' + msg))
        self.preview_worker = worker
        worker.finished.connect(self.preview_finished)

    def preview_finished(self):
        self.preview_worker = None
        # 参数变动发生在计算期间时，再计算最新快照，不堆积过时任务。
        if getattr(self, 'displayed_revision', -1) != self.preview_revision and not self.closing:
            self.preview_timer.start()

    def show_preview(self, payload):
        revision, original, processed, saturated, path = payload
        self.displayed_revision = revision
        if revision != self.preview_revision:
            return
        for array, label in [(original, self.original_preview), (processed, self.processed_preview)]:
            h, w, _ = array.shape
            image = QImage(array.data, w, h, array.strides[0], QImage.Format.Format_RGB888).copy()
            label.setPixmap(QPixmap.fromImage(image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
        self.preview_note.setText(f'{Path(path).name} · 原图近高光像素 {saturated:.1f}%（任一 RGB 通道 ≥250，仅作提示）'
                                 ' · 左：原图 / 右：缩略预览')

    def start_worker(self, fn, result, primary=True, error=None):
        worker = Worker(fn)
        self.workers.add(worker)
        worker.result.connect(result)
        worker.error.connect(error or self.show_error)
        worker.progress.connect(self.on_progress)
        worker.finished.connect(self.worker_finished)
        if primary:
            self.job = worker
            self.set_busy(True)
        worker.start()
        return worker

    def auto_detect_tools(self):
        """后台检测工具；保存过的有效路径优先，其次发现唯一路径才自动填入。"""
        if self.job or self.closing:
            return
        fields = self.tool_path_fields()
        current = {key: field.text().strip() for key, field in fields.items()}
        self._detection_snapshot = dict(current)
        # 保留文件夹对话框使用的 colmap.exe 当前值，以及仍然有效的路径。
        for key in KEYS:
            if is_tool_path(key, current[key]):
                current[key] = str(Path(current[key]).expanduser().resolve())
        self.detect_button.setEnabled(False)
        self._set_discovery_text('正在后台查找安装位置并检查 Python 依赖…')
        self.stage_label.setText('搜索本机已安装工具…')
        def scan(report, check):
            return discover(Path(__file__).resolve().parents[1], current, report, check)
        self.start_worker(scan, self.detection_finished, error=self.detection_failed)

    def detection_finished(self, found):
        self.detect_button.setEnabled(True)
        candidates = found['candidates']
        self.detected_tools = candidates
        self.candidates_button.setEnabled(any(candidates.values()))
        # 只自动填唯一候选；多个版本需要用户决定，不静默覆盖当前选择。
        for key, values in candidates.items():
            field = self.tool_path_fields()[key]
            value = field.text().strip()
            if value == self._detection_snapshot.get(key) and not is_tool_path(key, value) and len(values) == 1:
                field.setText(values[0])
        labels = {'vggt_python': 'VGGT Python', 'vggt_repo': 'VGGT 仓库',
                  'colmap': 'COLMAP', 'openmvs_bin': 'OpenMVS'}
        # 给用户看匹配来源/未发现原因和依赖检查结果，而不是把“路径存在”描述成“环境可运行”。
        summary, configured = [], {}
        for key, values in candidates.items():
            label = labels[key]
            field = self.tool_path_fields()[key]
            if is_tool_path(key, field.text().strip()):
                detail = found.get('python_checks', {}).get(str(Path(field.text()).resolve()), '')
                configured[label] = f'{field.text()} ({detail})' if detail else f'{field.text()} (程序文件已找到，运行环境待验证)'
                summary.append(f'{label}：已找到并配置')
            elif len(values) > 1:
                summary.append(f'{label}：找到 {len(values)} 个版本，点击“选择检测结果 / 版本”')
            elif not values:
                summary.append(f'{label}：未找到匹配项，可扫描指定目录')
        failed_python = [(path, reason) for path, reason in found.get('python_checks', {}).items()
                         if not reason.startswith('核心依赖存在')]
        unusable = [f'{path}：{reason}' for path, reason in failed_python]
        notes = list(found.get('notes', []))
        if unusable:
            notes.append(f'另有 {len(unusable)} 个 Python 候选缺少依赖或不可启动，完整详情见日志。')
        self.stage_label.setText('工具检测完成')
        self._set_discovery_text(' · '.join(summary + notes) or '未发现工具，请使用“扫描指定目录”。')
        self.log.appendPlainText('工具检测完成：' + (self.discovery_label.text()))
        for name, path in configured.items():
            self.log.appendPlainText(f'{name}：{path}')
        for path, message in found.get('python_checks', {}).items():
            self.log.appendPlainText(f'Python 环境检查：{path} · {message}')
        for item in unusable:
            self.log.appendPlainText('Python 候选不可用：' + item)
        self.log.appendPlainText(f'扫描目录数：{found.get("directories_scanned", 0)}')
        self.cache_tool_paths()

    def detection_failed(self, message):
        self.detect_button.setEnabled(True)
        self.stage_label.setText('工具检测失败')
        self._set_discovery_text('检测未完成：' + message)
        self.log.appendPlainText('自动检测失败：' + message)

    def _set_discovery_text(self, text):
        """Set tool-detection status and make the left scroll rail fit all lines."""
        self.discovery_label.setText(text)
        QTimer.singleShot(0, self._fit_discovery_label)

    def _fit_discovery_label(self):
        """Force the wrapped status label to its full height for its current width."""
        label = getattr(self, 'discovery_label', None)
        if label is None or label.width() < 2:
            return
        label.setMinimumHeight(0)
        measured = label.heightForWidth(label.contentsRect().width())
        if measured > 0:
            label.setMinimumHeight(measured + 4)
            label.updateGeometry()

    def choose_scan_roots(self):
        """用户明确选择目录后才扩大搜索范围。"""
        if self.job:
            return
        folders = QFileDialog.getExistingDirectory(self, '选择要扫描的安装/工作目录')
        if not folders:
            return
        fields = {'vggt_python': self.vggt_python, 'vggt_repo': self.vggt_repo,
                  'colmap': self.colmap, 'openmvs_bin': self.openmvs}
        current = {key: field.text().strip() if is_tool_path(key, field.text().strip()) else ''
                   for key, field in fields.items()}
        self._detection_snapshot = {key: field.text().strip() for key, field in fields.items()}
        self.detect_button.setEnabled(False)
        self.stage_label.setText('正在扫描所选目录…')
        def scan(report, check):
            return discover(Path(__file__).resolve().parents[1], current, report, check,
                            roots=[Path(folders)], scan_seconds=60, max_dirs=60000)
        self.start_worker(scan, self.detection_finished, error=self.detection_failed)

    def tool_path_fields(self):
        return {'vggt_python': self.vggt_python, 'vggt_repo': self.vggt_repo,
                'colmap': self.colmap, 'openmvs_bin': self.openmvs}

    def choose_detected_tools(self):
        if self.job:
            return
        labels = {'vggt_python': 'VGGT Python', 'vggt_repo': 'VGGT 仓库',
                  'colmap': 'COLMAP', 'openmvs_bin': 'OpenMVS'}
        keys = [key for key, values in self.detected_tools.items() if values]
        if not keys:
            return
        label, ok = QInputDialog.getItem(self, '选择检测结果', '选择要配置的工具：',
                                        [labels[key] for key in keys], 0, False)
        if not ok:
            return
        key = next(k for k in keys if labels[k] == label)
        choices = self.detected_tools[key]
        field = self.tool_path_fields()[key]
        index = choices.index(field.text()) if field.text() in choices else 0
        value, ok = QInputDialog.getItem(self, label, '选择安装版本：', choices, index, False)
        if ok:
            field.setText(value)
            self.cache_tool_paths()
            self.statusBar().showMessage(label + ' 已选择并记住，下次启动自动恢复。')

    def tool_cache_path(self):
        return Path(__file__).resolve().parents[1] / '.local' / 'tool_paths.json'

    def cache_tool_paths(self):
        try:
            path = self.tool_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_json(path, {'tools': {k: f.text().strip() for k, f in self.tool_path_fields().items()}})
        except OSError as exc:
            self.log.appendPlainText('工具路径无法记住：' + str(exc))

    def restore_tool_paths(self):
        path = self.tool_cache_path()
        if not path.is_file():
            return
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))['tools']
            for key, field in self.tool_path_fields().items():
                value = saved.get(key, '')
                if isinstance(value, str) and is_tool_path(key, value):
                    field.setText(value)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.log.appendPlainText('无法恢复上次工具路径，将重新检测：' + str(exc))

    def worker_finished(self):
        worker = self.sender()
        if worker is self.job:
            self.job = None
            self.set_busy(False)
        self.workers.discard(worker)
        worker.deleteLater()
        if self.closing and not self.workers:
            QTimer.singleShot(0, self.close)

    def set_busy(self, busy):
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        for method, viewer in self.viewers.items():
            viewer.eval_button.setEnabled(not busy and method in self.models)
        if not busy and self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        if not busy:
            self.stage_label.setText('就绪 · 详情见日志')

    def start(self):
        methods = [m for m, box in [('A', self.path_a), ('B', self.path_b)] if box.isChecked()]
        if len(self.drop.paths()) < 2 or not methods or not self.output.text().strip():
            QMessageBox.warning(self, '输入不完整', '重建需要至少两张照片、至少一条路径和结果保存目录。')
            return
        self.log.clear()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        for method in methods:
            self.eval_panels[method].reset('等待本次重建')
        task = reconstruction_task(self.drop.paths(), self.output.text().strip(), methods,
                                   self.prep_config(), self.tool_config(), self.api_config())
        self.start_worker(task, self.task_finished)
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage('任务使用启动时的配置快照；后续调参仅影响下一次任务。')

    def task_finished(self, result):
        manifest = result['manifest']
        self.last_run = result['run_dir']
        if manifest.get('status') in {'success', 'partial'} and self.pending_cleanup_run:
            if self.cleanup_previous_run(result['run_dir']):
                self.log.appendPlainText(f'已清理上一个任务目录：{self.pending_cleanup_run}')
                self.pending_cleanup_run = None
        self.save_session()
        self.log.appendPlainText('任务结束：' + manifest['status'] + ' · ' + result['run_dir'])
        if any(r['status'] == 'success' for r in manifest['results'].values()):
            self.tabs.setCurrentIndex(1)

    def cleanup_previous_run(self, current_run):
        """Remove only the prior app-created run, after a replacement succeeds."""
        try:
            old_candidate = Path(self.pending_cleanup_run).expanduser()
            if not old_candidate.exists():
                return True
            old = old_candidate.resolve(strict=True)
            current = Path(current_run).expanduser().resolve()
            if (old == current or not re.fullmatch(r'run_\d{8}_\d{6}_[0-9a-f]{8}', old.name)
                    or not (old / 'manifest.json').is_file()):
                raise ValueError('旧任务目录校验失败，已保留未删除。')
            manifest = json.loads((old / 'manifest.json').read_text(encoding='utf-8'))
            if not isinstance(manifest, dict) or 'input_images' not in manifest or 'results' not in manifest:
                raise ValueError('旧任务清单格式不符，已保留未删除。')
            old_resolved = old
            for method, model in list(self.models.items()):
                try:
                    model_path = Path(model).resolve()
                    if model_path.is_relative_to(old_resolved):
                        self.viewers[method].clear_model()
                        self.eval_panels[method].reset('上一任务结果已清理')
                        self.models.pop(method, None)
                        self.evaluations.pop(method, None)
                except (OSError, ValueError):
                    continue
            shutil.rmtree(old)
            return True
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.log.appendPlainText(f'未能自动清理上一个任务目录：{exc}')
            return False

    def cancel(self):
        if self.job:
            self.job.cancel()
            if self._active_vggt_dialog is not None:
                self._active_vggt_dialog.reject()
            self.cancel_button.setEnabled(False)
            self.stage_label.setText('正在取消…')
            self.log.appendPlainText('已请求取消。HTTP 等待中的取消将在响应返回或读取超时后生效。')

    def show_error(self, message):
        self.log.appendPlainText('任务信息：' + message)
        self.statusBar().showMessage(message)

    def on_progress(self, item):
        typ = item.get('type')
        if typ == 'log':
            self.log.appendPlainText(item['text'])
        elif typ == 'run':
            self.last_run = item['path']
            self.run_path.setText('任务输出：' + item['path'])
            self.save_session()
        elif typ == 'progress':
            value = item.get('value')
            self.progress.setRange(0, 0 if value is None else 100)
            if value is not None:
                self.progress.setValue(value)
            self.stage_label.setText(item.get('text', ''))
        elif typ == 'discovery_status':
            self.stage_label.setText(item.get('text', '搜索工具…'))
        elif typ == 'model':
            self.display_model(item['method'], item['path'])
        elif typ == 'vggt_preview':
            self.open_vggt_confidence_preview(item)
        elif typ == 'evaluation':
            self.evaluations[item['method']] = item['data']
            self.eval_panels[item['method']].display(item['data'])
        elif typ == 'evaluation_error':
            self.eval_panels[item['method']].reset('评价失败，可重试')
            self.eval_panels[item['method']].text.setPlainText(item['text'])
            self.log.appendPlainText('评价失败：' + item['text'])

    def load_model(self, method):
        if self.job:
            self.statusBar().showMessage('请等待当前任务结束后加载其他模型。')
            return
        path, _ = QFileDialog.getOpenFileName(self, '加载模型（贴图须保留相对路径）',
                                            self.output.text(), '三维模型 (*.obj *.ply)')
        if path:
            self.display_model(method, path)

    def display_model(self, method, path):
        self.models[method] = str(path)
        self.evaluations.pop(method, None)
        self.eval_panels[method].reset('尚未评价')
        try:
            self.viewers[method].load_model(path)
        except Exception as exc:
            self.log.appendPlainText(f'模型已保存，但显示失败：{exc} · {path}')
        self.viewers[method].eval_button.setEnabled(self.job is None)
        self.save_session()

    def open_vggt_confidence_preview(self, request):
        """Let the user compare confidence filters before committing the VGGT PLY."""
        path = Path(request['path'])
        try:
            with np.load(path) as archive:
                points = np.asarray(archive['points'], dtype=np.float32)
                colors = np.asarray(archive['colors'], dtype=np.uint8)
                confidence = np.asarray(archive['confidence'], dtype=np.float32)
            valid = np.isfinite(confidence) & np.isfinite(points).all(axis=1)
            points, colors, confidence = points[valid], colors[valid], confidence[valid]
            if not len(points):
                raise ValueError('VGGT 没有生成有效的置信度点云。')
            preview_count = min(len(points), 100000)
            preview_indices = np.linspace(0, len(points) - 1, preview_count, dtype=np.int64)
            preview_points = points[preview_indices]
            preview_colors = colors[preview_indices]
            preview_confidence = confidence[preview_indices]
            del points, colors
            dialog = QDialog(self)
            self._active_vggt_dialog = dialog
            dialog.setWindowTitle('VGGT 置信度预览 · 确认后保存')
            dialog.resize(980, 720)
            layout = QVBoxLayout(dialog)
            viewer = ModelViewer('拖动阈值，对比 VGGT 点云效果', dialog,
                                 auto_rotate=False, standalone=True)
            viewer.load_button.hide()
            viewer.eval_button.hide()
            viewer.reset_button.hide()
            viewer.zoom_button.hide()
            viewer.setMinimumHeight(430)
            layout.addWidget(viewer, 1)
            controls = QHBoxLayout()
            controls.addWidget(QLabel('置信度百分位'))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 99)
            slider.setValue(35)
            slider.setMinimumWidth(300)
            controls.addWidget(slider, 1)
            value_label = QLabel()
            controls.addWidget(value_label)
            layout.addLayout(controls)
            detail = QLabel(f'滑块越往右，过滤越严格。界面用均匀抽取的最多10万个点快速预览；确认后将阈值 {"按全部有效 VGGT 点的置信度分布计算"} 并应用到完整点云。')
            detail.setWordWrap(True)
            layout.addWidget(detail)
            buttons = QHBoxLayout()
            save_button = QPushButton('确认并保存所选效果')
            save_button.setObjectName('primaryButton')
            cancel_button = QPushButton('取消本次 VGGT')
            buttons.addStretch()
            buttons.addWidget(cancel_button)
            buttons.addWidget(save_button)
            layout.addLayout(buttons)
            selected = {'percentile': 35}

            def update_preview(value):
                threshold = (float(np.min(confidence)) if value == 0 else
                             float(np.percentile(confidence, value)))
                preview_mask = preview_confidence >= threshold
                full_kept = int(np.count_nonzero(confidence >= threshold))
                selected['percentile'] = value
                value_label.setText(f'阈值 {threshold:.3f} · 预览 {int(preview_mask.sum()):,}/{len(preview_mask):,} · 全量保留 {full_kept:,}（最多导出100,000）')
                viewer.set_point_cloud_data(preview_points[preview_mask], preview_colors[preview_mask])

            slider.valueChanged.connect(update_preview)
            update_preview(slider.value())
            save_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            self._active_vggt_dialog = None
            viewer.shutdown()
            request['percentile'] = selected['percentile'] if accepted else None
            request['cancelled'] = not accepted
            request['decision'].set()
            if accepted:
                self.statusBar().showMessage('已确认置信度效果，正在写入最终 VGGT 点云…')
            else:
                self.log.appendPlainText('已取消 VGGT 置信度预览；不会保存该路径模型。')
        except Exception as exc:
            self._active_vggt_dialog = None
            request['cancelled'] = True
            request['decision'].set()
            self.show_error(f'无法打开 VGGT 置信度预览，未保存模型：{exc}')

    def evaluate_model(self, method):
        if self.job or method not in self.models:
            return
        cfg = self.api_config()
        try:
            validate_api(cfg)
        except Exception as exc:
            QMessageBox.warning(self, 'API 配置不完整', str(exc))
            return
        model = self.models[method]
        def call(report, check):
            report({'type': 'progress', 'value': None, 'text': '上传模型 / 等待 API'})
            return evaluate(model, {'method': method, 'model_name': Path(model).name}, cfg, check)
        def done(payload):
            self.evaluations[method] = payload
            self.eval_panels[method].display(payload)
            self.save_session()
            self.log.appendPlainText(f'路径 {method} 评价完成。')
        self.start_worker(call, done, error=lambda msg: self.evaluation_failed(method, msg))

    def evaluation_failed(self, method, message):
        self.eval_panels[method].reset('评价失败，可重试')
        self.eval_panels[method].text.setPlainText(message)
        self.show_error(message)

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存非敏感配置', 'glacier_settings.json', 'JSON (*.json)')
        if path:
            try:
                # API 地址和密钥都不落盘，避免 URL 中的 token 被无意持久化。
                save_json(path, {'version': 1, 'output': self.output.text(), 'tools': asdict(self.tool_config()),
                                 'preprocessing': asdict(self.prep_config())})
                self.statusBar().showMessage('工具和预处理配置已保存；API 地址和密钥未写入文件。')
            except Exception as exc:
                QMessageBox.warning(self, '保存失败', str(exc))

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载配置', '', 'JSON (*.json)')
        if not path:
            return
        try:
            cfg = json.loads(Path(path).read_text(encoding='utf-8'))
            tools = PipelineConfig(**cfg.get('tools', {}))
            prep = PreprocessConfig(**cfg.get('preprocessing', {}))
            fields = {'vggt_python': self.vggt_python, 'vggt_repo': self.vggt_repo, 'vggt_args': self.vggt_args,
                      'colmap': self.colmap, 'openmvs_bin': self.openmvs,
                      'colmap_extras': self.colmap_extra, 'openmvs_extras': self.openmvs_extra}
            for key, value in asdict(tools).items():
                fields[key].setText(str(value))
            for key, value in asdict(prep).items():
                self.prep_widgets[key].setValue(value)
            self.output.setText(str(cfg.get('output', self.output.text())))
            self.cache_tool_paths()
        except Exception as exc:
            QMessageBox.warning(self, '配置加载失败', str(exc))

    def closeEvent(self, event):
        if self.workers:
            # 不强行 terminate QThread；先结束子进程/请求，再安全关闭 VTK。
            self.closing = True
            self.preview_timer.stop()
            for worker in self.workers:
                worker.cancel()
            self.stage_label.setText('等待后台退出后关闭…')
            event.ignore()
            return
        for viewer in self.viewers.values():
            viewer.shutdown()
        self.cache_tool_paths()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'discovery_label'):
            QTimer.singleShot(0, self._fit_discovery_label)
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(
                'font-size:28px;font-weight:700;color:#FFFFFF;' if self.width() < 1180
                else 'font-size:28px;font-weight:700;color:#FFFFFF;')


STYLE = """
QWidget { font-family:'Segoe UI','PingFang SC','Microsoft YaHei UI','Microsoft YaHei',sans-serif; font-size:14px; color:#F1F6FC; background:transparent; }
QMainWindow { background:#05080E; }
QScrollArea, QScrollArea > QWidget > QWidget { background:transparent; }
QSplitter::handle { background:transparent; }
QGroupBox { font-size:18px; font-weight:900; border:0; border-radius:10px; margin-top:12px; padding:8px 2px 8px; background:transparent; }
QGroupBox::title { subcontrol-origin:margin; left:32px; padding:0; color:#CEC8BB; font-family:'Noto Sans SC','Microsoft YaHei UI','Microsoft YaHei','Segoe UI',sans-serif; font-size:18px; font-weight:900; }
QLineEdit, QPlainTextEdit, QListWidget, QSpinBox, QDoubleSpinBox, QTableWidget, QComboBox { background:rgb(13,20,32); border:1px solid rgb(72,86,104); border-radius:10px; padding:3px 7px; color:#FFFFFF; selection-background-color:#168FEA; alternate-background-color:rgb(21,30,44); }
QLineEdit#pathField { font-size:11px; padding-left:5px; padding-right:5px; }
QListWidget#glassDropList, QListWidget#glassDropList::viewport { background:transparent; border:0; }
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border:1px solid #168FEA; }
QListWidget::item { padding:7px 8px; border-radius:8px; }
QListWidget::item:selected, QTableWidget::item:selected { background:rgb(45,67,112); color:#FFFFFF; }
QLabel#viewerHint { color:rgba(235,244,252,220); padding:5px 2px 0; }
QLabel#viewerStatus { color:rgba(205,218,232,180); font-size:12px; padding:0 2px 3px; }
QHeaderView::section { background:rgb(24,33,48); color:#CEC8BB; border:0; padding:8px; }
QPushButton { min-height:20px; background:rgb(17,24,38); color:#F1F6FC; border:1px solid rgb(94,108,126); border-radius:18px; padding:4px 12px; font-size:12px; }
QPushButton#imageActionButton { background:transparent; border:0; border-radius:0; padding:4px 16px; min-height:30px; color:#F5FAFF; }
QPushButton#browseButton { min-height:24px; padding:2px 4px; font-size:12px; border-radius:15px; background:rgb(17,24,38); border:1px solid rgb(94,108,126); }
QPushButton#browseButton:hover { background:rgb(30,42,60); border-color:rgb(150,168,188); }
QPushButton:hover { background:rgb(30,42,60); border-color:rgb(150,168,188); }
QPushButton:pressed { background:rgb(38,48,63); }
QPushButton:disabled { color:rgb(128,140,154); background:rgb(15,21,31); border-color:rgb(54,64,78); }
QPushButton#primaryButton { background:rgb(24,32,48); color:#FFFFFF; border:1px solid rgb(126,144,164); font-weight:600; padding:8px 24px; }
QPushButton#primaryButton:hover { background:rgb(40,52,73); border-color:rgb(180,194,208); }
QTabWidget::pane { border:0; background:transparent; top:-1px; }
QTabWidget#glassTabs::pane { border:0; background:transparent; }
QScrollArea#glassRail, QScrollArea#resultsScroll { border:0; background:transparent; }
QTabBar::tab { font-family:'Source Han Sans CN','思源黑体','Noto Sans CJK SC','Microsoft YaHei UI','Microsoft YaHei','Segoe UI',sans-serif; font-size:16px; font-weight:800; color:transparent; background:transparent; border:0; padding:8px 18px; margin:2px; }
QProgressBar { border:1px solid rgb(94,108,126); border-radius:100px; text-align:center; min-height:16px; background:rgb(13,20,32); color:#FFFFFF; }
QProgressBar::chunk { background:#AEC7D9; border-radius:100px; }
QCheckBox { spacing:10px; }
QCheckBox::indicator { width:18px; height:18px; border-radius:5px; border:1px solid rgb(120,136,154); background:rgb(13,20,32); }
QCheckBox::indicator:checked { background:#168FEA; border-color:#168FEA; }
QScrollBar:vertical { background:transparent; width:10px; margin:4px; }
QScrollBar::handle:vertical { background:rgba(255,255,255,60); border-radius:5px; min-height:28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QStatusBar { font-family:'Source Han Sans CN','思源黑体','Noto Sans CJK SC','Microsoft YaHei UI','Microsoft YaHei','Segoe UI',sans-serif; font-size:11px; font-weight:700; color:#95A6B8; background:transparent; }
QToolTip { color:#FFFFFF; background:#151923; border:1px solid rgba(255,255,255,65); border-radius:8px; padding:6px; }
QLabel#appTitle { font-size:28px; font-weight:700; color:#FFFFFF; }
QLabel#subtitle { color:rgba(220,234,248,175); font-size:12px; }
QLabel#eyebrow { color:#D2DFEB; font-size:20px; font-weight:700; letter-spacing:2px; }
QLabel#statusText { font-family:'Source Han Sans CN','思源黑体','Noto Sans CJK SC','Microsoft YaHei UI','Microsoft YaHei','Segoe UI',sans-serif; color:#C7D8E8; font-size:11px; font-weight:700; }
QLabel#previewLabel { font-family:'Source Han Sans CN','思源黑体','Noto Sans CJK SC','Microsoft YaHei UI','Microsoft YaHei','Segoe UI',sans-serif; font-weight:700; }
QLabel#sectionTitle { font-size:16px; font-weight:600; color:#FFFFFF; }
QLabel#hint { color:rgba(215,230,245,155); font-size:11px; }
"""


def run():
    # Windows otherwise identifies the running window through pythonw.exe,
    # which can leave its taskbar button with the interpreter's blank icon.
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'GlacierReconstruction.DesktopApp')
        except (AttributeError, OSError):
            pass
    app = QApplication.instance() or QApplication([])
    app.setApplicationName('冰川三维重建')
    app.setOrganizationName('GlacierReconstruction')
    icon_path = Path(__file__).resolve().parent.parent / 'assets' / 'glacier_icon.ico'
    if icon_path.is_file():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
        app.setProperty('glacier_icon_path', str(icon_path))
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
