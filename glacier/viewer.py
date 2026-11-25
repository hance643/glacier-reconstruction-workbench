"""原生 VTK 双视口模型显示，OBJ 使用 importer 保留 MTL 及多张贴图。

PLY 是彩色点云或网格；没有 UV/贴图的点云只显示顶点色。两个相机独立，
因为两种算法不保证同一坐标系/尺度，框架不伪造几何对齐。
"""
from pathlib import Path
import shlex
import struct

import numpy as np
from PySide6.QtCore import Signal, Qt, QEvent, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QCheckBox, QDialog, QSizePolicy, QComboBox, QSlider)

try:
    # 后端注册必须显式导入，否则只有抽象渲染对象。
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
    # vtkPoints is defined in vtkCommonCore; vtkCellArray/PolyData belong to
    # vtkCommonDataModel in the VTK 9.7 wheels used by the bundled environment.
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.vtkIOPLY import vtkPLYReader
    from vtkmodules.vtkIOImport import vtkOBJImporter
    # vtkVertexGlyphFilter 属于 vtkFiltersGeneral；导错模块会触发
    # ImportError，之前被下面的 fallback 捕获后误报成“未安装 VTK”。
    from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError as exc:
    VTK_AVAILABLE = False
    VTK_IMPORT_ERROR = str(exc)


if VTK_AVAILABLE:
    class PreviewVTKWidget(QVTKRenderWindowInteractor):
        """Route embedded-preview wheel input to camera zoom, not VTK pan/scroll."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.preview_wheel_handler = None

        def wheelEvent(self, event):
            if self.preview_wheel_handler is not None:
                self.preview_wheel_handler(event.angleDelta().y(), event.pixelDelta().y())
                event.accept()
                return
            super().wheelEvent(event)


class ModelViewer(QWidget):
    load_requested = Signal()
    evaluate_requested = Signal()

    def __init__(self, title, parent=None, *, auto_rotate=True, standalone=False, vggt_controls=False):
        super().__init__(parent)
        self.title = title
        self.auto_rotate = auto_rotate
        self.standalone = standalone
        self.vggt_controls = vggt_controls
        self.path = None
        self.scroll_area = None
        self.actors = []
        self.textures = []
        self.hold = []
        self._large_dialog = None
        self._vggt_source_poly = None
        self._vggt_points = None
        self._vggt_colors = None
        self._vggt_confidence = None
        self._vggt_frame_ids = None
        self.camera_actor = None
        self._filtered_glyph = None
        self._vggt_control_row = None
        self.spin_timer = QTimer(self)
        self.spin_timer.setInterval(60)
        self.spin_timer.timeout.connect(self.rotate_preview)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setObjectName('sectionTitle')
        layout.addWidget(heading)
        row = QHBoxLayout()
        load = QPushButton('加载 OBJ / PLY')
        load.clicked.connect(self.load_requested.emit)
        row.addWidget(load)
        self.eval_button = QPushButton('评价 / 重试')
        self.eval_button.setEnabled(False)
        self.eval_button.clicked.connect(self.evaluate_requested.emit)
        row.addWidget(self.eval_button)
        reset = QPushButton('重置视角')
        reset.clicked.connect(self.reset_camera)
        row.addWidget(reset)
        if standalone:
            load.hide()
            self.eval_button.hide()
        layout.addLayout(row)
        if standalone and vggt_controls:
            self._add_vggt_controls(layout)
        options = QHBoxLayout()
        self.texture_toggle = QCheckBox('纹理 / 顶点颜色')
        self.texture_toggle.setChecked(True)
        self.texture_toggle.toggled.connect(self.update_style)
        self.wireframe = QCheckBox('线框')
        self.wireframe.toggled.connect(self.update_style)
        options.addWidget(self.texture_toggle)
        options.addWidget(self.wireframe)
        options.addStretch()
        layout.addLayout(options)
        self.caption = QLabel('等待加载模型')
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.TextFormat.PlainText)
        self.caption.setObjectName('viewerStatus')
        if VTK_AVAILABLE:
            self.canvas = PreviewVTKWidget(self)
            if not standalone:
                self.canvas.preview_wheel_handler = self.handle_preview_wheel
            self.canvas.setMinimumSize(250, 204 if not standalone else 300)
            if not standalone:
                # Three fifths of the previous 340px embedded viewport. The
                # separate zoom window remains unrestricted for close study.
                self.canvas.setMaximumHeight(204)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.canvas.installEventFilter(self)
            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.035, 0.040, 0.055)
            self.renderer.SetBackground2(0.055, 0.075, 0.115)
            self.renderer.GradientBackgroundOn()
            self.canvas.GetRenderWindow().AddRenderer(self.renderer)
            self.canvas.Initialize()
            layout.addWidget(self.canvas, 1)
        else:
            detail = globals().get('VTK_IMPORT_ERROR', '未知导入错误')
            label = QLabel('VTK 三维视图无法加载：可以导入图像、重建与调用 API，\n'
                           '请检查 VTK 安装或模块依赖。\n\n'
                           f'诊断信息：{detail}')
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 1)

        # Keep all explanatory/model metadata in a normal-flow area directly
        # after the render canvas, without a panel background or fixed height.
        self.footer_panel = QWidget(self)
        footer = QHBoxLayout(self.footer_panel)
        footer.setContentsMargins(0, 4, 0, 0)
        footer_text = QVBoxLayout()
        self.hint = QLabel('预览自动旋转 · 放大查看后可用鼠标旋转、滚轮缩放、Shift+左键平移')
        self.hint.setWordWrap(True)
        self.hint.setObjectName('viewerHint')
        footer_text.addWidget(self.hint)
        footer_text.addWidget(self.caption)
        footer.addLayout(footer_text, 1)
        self.zoom_button = QPushButton('放大查看')
        self.zoom_button.setObjectName('primaryButton')
        self.zoom_button.setEnabled(False)
        self.zoom_button.clicked.connect(self.show_large_view)
        self.zoom_button.setVisible(not standalone)
        footer.addWidget(self.zoom_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.footer_panel, 0)

    def eventFilter(self, watched, event):
        if (VTK_AVAILABLE and watched is self.canvas
                and not self.standalone
                and event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton):
            self.show_large_view()
            return True
        return super().eventFilter(watched, event)

    def handle_preview_wheel(self, angle_delta, pixel_delta=0):
        if self.path:
            self.zoom_camera(angle_delta, pixel_delta)
        elif self.scroll_area is not None:
            bar = self.scroll_area.verticalScrollBar()
            if bar.maximum() > bar.minimum():
                delta = angle_delta or pixel_delta
                bar.setValue(bar.value() - delta)

    def zoom_camera(self, angle_delta, pixel_delta=0):
        if not VTK_AVAILABLE or not self.path:
            return
        delta = angle_delta or pixel_delta
        if not delta:
            return
        camera = self.renderer.GetActiveCamera()
        steps = max(-10, min(10, delta / (120 if angle_delta else 40)))
        factor = 1.15 ** steps
        parallel = camera.GetParallelProjection()
        current = camera.GetParallelScale() if parallel else camera.GetDistance()
        distance = current / factor
        # Keep the camera within a useful range for both small and large models.
        bounds = self.renderer.ComputeVisiblePropBounds()
        if bounds[0] < bounds[1]:
            model_size = max(bounds[i + 1] - bounds[i] for i in (0, 2, 4))
            distance = min(max(distance, model_size * 0.005), model_size * 100)
        # SetDistance moves the focal point, which changes the orbit centre
        # rather than the model's projected size. Dolly preserves that centre.
        if parallel:
            camera.SetParallelScale(distance)
        elif distance > 0:
            camera.Dolly(current / distance)
        self.renderer.ResetCameraClippingRange()
        self.canvas.GetRenderWindow().Render()

    def show_large_view(self):
        """Open a separate interactive VTK window; leave this preview spinning."""
        if not self.path:
            return
        if self._large_dialog is not None:
            self._large_dialog.showNormal()
            self._large_dialog.raise_()
            self._large_dialog.activateWindow()
            return
        dialog = QDialog(self.window())
        dialog.setWindowTitle(f'{self.title} · 放大查看')
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        detail = ModelViewer(self.title, dialog, auto_rotate=False, standalone=True,
                             vggt_controls=self.vggt_controls)
        try:
            detail.load_model(self.path)
        except Exception as exc:
            detail.caption.setText(f'模型加载失败：{exc}')
        body = QVBoxLayout(dialog)
        body.addWidget(detail)
        available = dialog.screen().availableGeometry()
        width = min(1500, max(800, int(available.width() * .78)))
        height = min(1000, max(600, int(available.height() * .78)))
        dialog.resize(width, height)
        frame = dialog.frameGeometry()
        frame.moveCenter(available.center())
        dialog.move(frame.topLeft())
        dialog.finished.connect(lambda *_: detail.shutdown())
        dialog.finished.connect(lambda *_: setattr(self, '_large_dialog', None))
        self._large_dialog = dialog
        dialog.show()

    def _add_vggt_controls(self, layout):
        row = QHBoxLayout()
        self.camera_toggle = QCheckBox('Show Cameras')
        self.camera_toggle.setChecked(True)
        self.camera_toggle.toggled.connect(self._toggle_cameras)
        row.addWidget(self.camera_toggle)
        row.addSpacing(12)
        row.addWidget(QLabel('Confidence Percent'))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setMinimumWidth(120)
        self.confidence_slider.setEnabled(False)
        self.confidence_label = QLabel('不可用')
        row.addWidget(self.confidence_slider, 1)
        row.addWidget(self.confidence_label)
        row.addSpacing(12)
        row.addWidget(QLabel('Show Points from Frames'))
        self.frame_selector = QComboBox()
        self.frame_selector.addItem('All', None)
        self.frame_selector.setMinimumWidth(90)
        self.frame_selector.currentIndexChanged.connect(self._update_vggt_point_filter)
        row.addWidget(self.frame_selector)
        layout.insertLayout(1, row)

    def _toggle_cameras(self, visible):
        if self.camera_actor is not None:
            self.camera_actor.SetVisibility(visible)
            self.canvas.GetRenderWindow().Render()

    @staticmethod
    def _read_colmap_images(path):
        """Parse image poses and their zero-based frame indices from COLMAP binary data."""
        data = Path(path).read_bytes()
        count = struct.unpack_from('<Q', data, 0)[0]
        offset = 8
        result = {}
        for _ in range(count):
            image_id = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            q = np.asarray(struct.unpack_from('<4d', data, offset), dtype=float)
            offset += 32
            t = np.asarray(struct.unpack_from('<3d', data, offset), dtype=float)
            offset += 28
            end = data.index(b'\0', offset)
            name = data[offset:end].decode('utf-8', errors='replace')
            offset = end + 1
            point_count = struct.unpack_from('<Q', data, offset)[0]
            offset += 8 + point_count * 24
            try:
                frame = int(Path(name).stem)
            except ValueError:
                frame = len(result)
            qw, qx, qy, qz = q
            rotation = np.asarray([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
            ])
            result[image_id] = (frame, rotation, t)
        return result

    def _read_vggt_frame_mapping(self, model_path, vertices):
        sparse = model_path.parent.parent / 'scene' / 'sparse'
        image_file, point_file = sparse / 'images.bin', sparse / 'points3D.bin'
        if not image_file.is_file() or not point_file.is_file():
            return None, None
        try:
            images = self._read_colmap_images(image_file)
            data = point_file.read_bytes()
            offset = 8
            point_count = struct.unpack_from('<Q', data, 0)[0]
            point_tracks = {}
            for _ in range(point_count):
                offset += 8
                xyz = struct.unpack_from('<3d', data, offset)
                offset += 35
                track_count = struct.unpack_from('<Q', data, offset)[0]
                offset += 8
                frames = set()
                for _ in range(track_count):
                    image_id, _point2d = struct.unpack_from('<II', data, offset)
                    offset += 8
                    if image_id in images:
                        frames.add(images[image_id][0])
                point_tracks[tuple(round(v, 4) for v in xyz)] = frames
            frame_ids = np.full(len(vertices), -1, dtype=np.int32)
            for index, vertex in enumerate(vertices):
                frames = point_tracks.get(tuple(round(float(v), 4) for v in vertex), ())
                if frames:
                    frame_ids[index] = min(frames)
            return images, frame_ids
        except (OSError, ValueError, IndexError, struct.error):
            return None, None

    def _add_vggt_cameras(self, images):
        if not images:
            self.camera_toggle.setEnabled(False)
            return
        bounds = self._vggt_source_poly.GetBounds()
        span = np.asarray([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        scale = max(float(np.linalg.norm(span)) * .035, .01)
        vtk_points, lines = vtkPoints(), vtkCellArray()

        def add_line(a, b):
            first = vtk_points.InsertNextPoint(*map(float, a))
            second = vtk_points.InsertNextPoint(*map(float, b))
            lines.InsertNextCell(2)
            lines.InsertCellPoint(first)
            lines.InsertCellPoint(second)

        for _frame, rotation, translation in images.values():
            center = -rotation.T @ translation
            local = np.asarray([[-scale,-scale,scale],[scale,-scale,scale],
                                [scale,scale,scale],[-scale,scale,scale]])
            corners = [rotation.T @ (corner - translation) for corner in local]
            for corner in corners:
                add_line(center, corner)
            for index in range(4):
                add_line(corners[index], corners[(index+1) % 4])
        geometry = vtkPolyData()
        geometry.SetPoints(vtk_points)
        geometry.SetLines(lines)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(geometry)
        self.camera_actor = vtkActor()
        self.camera_actor.SetMapper(mapper)
        self.camera_actor.GetProperty().SetColor(.09, .56, .95)
        self.camera_actor.GetProperty().SetLineWidth(2)
        self.camera_actor.SetVisibility(self.camera_toggle.isChecked())
        self.renderer.AddActor(self.camera_actor)

    def _update_vggt_point_filter(self, *_):
        if not (VTK_AVAILABLE and self._vggt_points is not None and self.actors):
            return
        frame = self.frame_selector.currentData()
        if frame is None or self._vggt_frame_ids is None:
            mask = np.ones(len(self._vggt_points), dtype=bool)
        else:
            mask = self._vggt_frame_ids == frame
        if self._vggt_confidence is not None and self.confidence_slider.isEnabled():
            threshold = np.percentile(self._vggt_confidence, self.confidence_slider.value())
            mask &= self._vggt_confidence >= threshold
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(self._vggt_points[mask]), deep=True))
        poly = vtkPolyData()
        poly.SetPoints(vtk_points)
        if self._vggt_colors is not None:
            colors = numpy_to_vtk(np.ascontiguousarray(self._vggt_colors[mask]), deep=True)
            colors.SetName('RGB')
            poly.GetPointData().SetScalars(colors)
        self._filtered_glyph = vtkVertexGlyphFilter()
        self._filtered_glyph.SetInputData(poly)
        self._filtered_glyph.Update()
        self.actors[0].GetMapper().SetInputConnection(self._filtered_glyph.GetOutputPort())
        self.canvas.GetRenderWindow().Render()

    def _prepare_vggt_controls(self, model_path, poly):
        self._vggt_source_poly = poly
        self._vggt_points = vtk_to_numpy(poly.GetPoints().GetData()).copy()
        colors = poly.GetPointData().GetScalars()
        if colors is not None and colors.GetNumberOfComponents() >= 3:
            self._vggt_colors = vtk_to_numpy(colors).copy()
        images, frame_ids = self._read_vggt_frame_mapping(model_path, self._vggt_points)
        self._vggt_frame_ids = frame_ids
        if frame_ids is not None:
            frames = sorted(int(frame) for frame in np.unique(frame_ids) if frame >= 0)
            self.frame_selector.blockSignals(True)
            self.frame_selector.clear()
            self.frame_selector.addItem('All', None)
            for frame in frames:
                self.frame_selector.addItem(str(frame), frame)
            self.frame_selector.blockSignals(False)
        else:
            self.frame_selector.setEnabled(False)
        if images:
            self._add_vggt_cameras(images)
        else:
            self.camera_toggle.setEnabled(False)

    def rotate_preview(self):
        if not (VTK_AVAILABLE and self.auto_rotate and self.path):
            return
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(0.9)
        self.renderer.ResetCameraClippingRange()
        self.canvas.GetRenderWindow().Render()

    def showEvent(self, event):
        super().showEvent(event)
        if self.auto_rotate and self.path and not self.spin_timer.isActive():
            self.spin_timer.start()

    def hideEvent(self, event):
        self.spin_timer.stop()
        super().hideEvent(event)

    def load_model(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() not in {'.ply', '.obj'} or not path.is_file():
            raise ValueError('请选择存在的 OBJ 或 PLY 模型。')
        if self._large_dialog is not None:
            self._large_dialog.close()
            self._large_dialog = None
        if not VTK_AVAILABLE:
            self.path = path
            self.eval_button.setEnabled(True)
            self.zoom_button.setEnabled(True)
            self.caption.setText(str(path))
            return
        self.renderer.RemoveAllViewProps()
        self.hold = []
        self._vggt_source_poly = None
        self._vggt_points = None
        self._vggt_colors = None
        self._vggt_confidence = None
        self._vggt_frame_ids = None
        self.camera_actor = None
        if self.standalone and self.vggt_controls:
            self.camera_toggle.setEnabled(True)
            self.frame_selector.setEnabled(True)
            self.frame_selector.clear()
            self.frame_selector.addItem('All', None)
        warning = ''
        if path.suffix.lower() == '.obj':
            importer = vtkOBJImporter()
            importer.SetFileName(str(path))
            # OBJ 的 mtllib 可与模型异名；优先读取文件里的相对引用。
            material = path.with_suffix('.mtl')
            with path.open(encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    if line.startswith('mtllib '):
                        name = line[len('mtllib '):].strip()
                        if (path.parent / name).is_file():
                            material = path.parent / name
                        else:
                            names = shlex.split(name)
                            if names:
                                material = path.parent / names[0]
                        break
            if material.is_file():
                importer.SetFileNameMTL(str(material))
                importer.SetTexturePath(str(material.parent))
            else:
                warning = ' · 未找到 MTL，显示几何模型'
            importer.SetRenderWindow(self.canvas.GetRenderWindow())
            importer.Update()
            self.hold.append(importer)
        else:
            reader = vtkPLYReader()
            reader.SetFileName(str(path))
            reader.Update()
            poly = reader.GetOutput()
            if poly.GetNumberOfPoints() == 0:
                raise ValueError('PLY 文件无有效顶点，无法显示。')
            mapper = vtkPolyDataMapper()
            if poly.GetNumberOfCells() == 0:
                glyph = vtkVertexGlyphFilter()
                glyph.SetInputData(poly)
                glyph.Update()
                mapper.SetInputConnection(glyph.GetOutputPort())
                self.hold.append(glyph)
            else:
                mapper.SetInputData(poly)
            mapper.SetColorModeToDirectScalars()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(2)
            self.renderer.AddActor(actor)
            self.hold.append(reader)
            warning = ' · PLY 顶点色；纹理网格请加载 OBJ + MTL'
            if self.standalone and self.vggt_controls:
                self._prepare_vggt_controls(path, poly)
        collection = self.renderer.GetActors()
        collection.InitTraversal()
        self.actors = [collection.GetNextActor() for _ in range(collection.GetNumberOfItems())]
        self.actors = [actor for actor in self.actors if actor is not self.camera_actor]
        points, faces = 0, 0
        for actor in self.actors:
            mapper = actor.GetMapper()
            if mapper:
                mapper.Update()
                data = mapper.GetInput()
                if data:
                    points += data.GetNumberOfPoints()
                    faces += data.GetNumberOfPolys()
        if not points:
            raise ValueError('模型中没有可显示的几何数据。')
        self.textures = [actor.GetTexture() for actor in self.actors]
        self.path = path
        self.eval_button.setEnabled(True)
        self.zoom_button.setEnabled(True)
        self.caption.setText(f'{path.name} · {points:,} 顶点 / {faces:,} 面{warning}')
        self.caption.setToolTip(str(path))
        self.update_style()
        self.reset_camera()
        if self.auto_rotate and self.isVisible() and not self.spin_timer.isActive():
            self.spin_timer.start()

    def update_style(self):
        if not VTK_AVAILABLE:
            return
        for actor, texture in zip(self.actors, self.textures):
            actor.SetTexture(texture if self.texture_toggle.isChecked() else None)
            actor.GetMapper().SetScalarVisibility(self.texture_toggle.isChecked())
            prop = actor.GetProperty()
            prop.SetColor(0.81, 0.80, 0.75)
            if self.wireframe.isChecked():
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()
        self.canvas.GetRenderWindow().Render()

    def reset_camera(self):
        if VTK_AVAILABLE:
            self.renderer.ResetCamera()
            self.canvas.GetRenderWindow().Render()

    def clear_model(self):
        if self._large_dialog is not None:
            self._large_dialog.close()
            self._large_dialog = None
        self.path = None
        self.actors = []
        self.textures = []
        self.hold = []
        self.eval_button.setEnabled(False)
        self.zoom_button.setEnabled(False)
        self.caption.setText('等待加载模型')
        self.caption.setToolTip('')
        if VTK_AVAILABLE:
            self.renderer.RemoveAllViewProps()
            self.canvas.GetRenderWindow().Render()

    def shutdown(self):
        self.spin_timer.stop()
        if self._large_dialog is not None:
            self._large_dialog.close()
            self._large_dialog = None
        if VTK_AVAILABLE:
            self.canvas.Finalize()
