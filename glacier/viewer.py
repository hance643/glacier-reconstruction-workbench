"""原生 VTK 双视口模型显示，OBJ 使用 importer 保留 MTL 及多张贴图。

PLY 是彩色点云或网格；没有 UV/贴图的点云只显示顶点色。两个相机独立，
因为两种算法不保证同一坐标系/尺度，框架不伪造几何对齐。
"""
from pathlib import Path
import shlex
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox

try:
    # 后端注册必须显式导入，否则只有抽象渲染对象。
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
    from vtkmodules.vtkIOPLY import vtkPLYReader
    from vtkmodules.vtkIOImport import vtkOBJImporter
    # vtkVertexGlyphFilter 属于 vtkFiltersGeneral；导错模块会触发
    # ImportError，之前被下面的 fallback 捕获后误报成“未安装 VTK”。
    from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
    VTK_AVAILABLE = True
except ImportError as exc:
    VTK_AVAILABLE = False
    VTK_IMPORT_ERROR = str(exc)


class ModelViewer(QWidget):
    load_requested = Signal()
    evaluate_requested = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.path = None
        self.actors = []
        self.textures = []
        self.hold = []
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
        layout.addLayout(row)
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
        self.caption = QLabel('等待模型 · 鼠标左键旋转 / 滚轮缩放 / Shift+左键平移')
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.TextFormat.PlainText)
        if VTK_AVAILABLE:
            self.canvas = QVTKRenderWindowInteractor(self)
            self.canvas.setMinimumSize(250, 280)
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
        layout.addWidget(self.caption)

    def load_model(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() not in {'.ply', '.obj'} or not path.is_file():
            raise ValueError('请选择存在的 OBJ 或 PLY 模型。')
        if not VTK_AVAILABLE:
            self.path = path
            self.eval_button.setEnabled(True)
            self.caption.setText(str(path))
            return
        self.renderer.RemoveAllViewProps()
        self.hold = []
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
        collection = self.renderer.GetActors()
        collection.InitTraversal()
        self.actors = [collection.GetNextActor() for _ in range(collection.GetNumberOfItems())]
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
        self.caption.setText(f'{path.name} · {points:,} 顶点 / {faces:,} 面{warning}')
        self.caption.setToolTip(str(path))
        self.update_style()
        self.reset_camera()

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

    def shutdown(self):
        if VTK_AVAILABLE:
            self.canvas.Finalize()
