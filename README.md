# 冰川三维重建可视化演示

这是一个上层调度 GUI：它不修改 VGGT、COLMAP 或 OpenMVS 的算法内核。程序负责读取照片、保留原始影像、生成统一的预处理副本、启动外部命令、保存独立任务目录、显示模型，并通过 HTTP 把模型压缩包送到用户自研评价 API。

## 技术选型

- Python 3.11+；PySide6 负责桌面 GUI、拖放、后台 QThread 和进度信号。
- Pillow / OpenCV 负责中文路径图像读取和 Lab 亮度通道处理。预处理包括高光压缩、CLAHE 局部对比度、Gamma、双边降噪和可选锐化。
- VTK 负责 OBJ/MTL/纹理和 PLY 顶点色显示；如果环境尚未安装 VTK，重建和评价仍可运行，但显示区只会提示模型路径。
- subprocess + psutil 调用外部工具；每个阶段的 stdout/stderr 进入任务日志，取消时连同子进程树一起退出。
- Requests Toolbelt 以 multipart 流式上传模型 ZIP，客户端不计算分数，只展示后端 JSON。

## 安装和运行

已配置本机快捷方式：双击上一级“冰川重建”目录中的“启动冰川三维重建.lnk”，即可使用项目的虚拟环境打开 GUI，无需输入命令，也不会弹出终端。可右键该快捷方式，选择“发送到 → 桌面快捷方式”。如果移动项目文件夹，需要更新快捷方式的目标路径和起始位置。启动错误会显示提示，日志保存在 glacier_demo/logs 中。

    cd glacier_demo
    python -m venv .venv
    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
    .venv\\Scripts\\python.exe main.py

程序启动后会在后台尝试查找 VGGT、COLMAP、OpenMVS 和 Python 环境。查找范围包括 PATH、VGGT_HOME / COLMAP_HOME / OPENMVS_HOME / CONDA_PREFIX 等环境变量、当前项目、常见用户安装目录及磁盘根目录下名称相关的安装目录；不会漫无边际地递归整个磁盘，也不会下载或安装工具。唯一候选会自动填入；多个版本可以在“选择检测结果 / 版本”中选择。

如果工具装在非标准位置，点“扫描指定目录”并选择它的父文件夹。VGGT Python 候选会做隔离模式的短时核心模块检查，但不会导入模型、不分配显存，不能保证权重、CUDA 或所有运行依赖已配置。COLMAP/OpenMVS 的可执行文件被找到，只代表文件可用；正式任务启动前会验证路径，运行时还会实际检查返回状态。检测记录显示在窗口“工具检测结果”区域和任务日志中。

工具路径在检测/手动选定后会记入 glacier_demo/.local/tool_paths.json，启动时只恢复仍然有效的路径。保存配置仍可用于备份工具路径、附加参数和预处理设置；API 密钥不会写入配置文件。VGGT Python 指向能运行 VGGT 的解释器，仓库目录必须含 demo_colmap.py；COLMAP 选择 colmap.exe 或 PATH 中的 colmap；OpenMVS bin 目录应包含 InterfaceCOLMAP、DensifyPointCloud、ReconstructMesh、RefineMesh 和 TextureMesh。

附加参数使用 JSON 数组或对象。程序保留并管理输入、输出和工作目录参数，避免附加参数把结果写到用户未选择的路径。

## 任务目录

每次任务创建 输出目录/run_时间_UUID/，包含预处理 images、A 和 B 的中间结果、manifest.json、task.log，以及可选的 A/evaluation.json 和 B/evaluation.json。原始图像只被读取。冰雪高光如果已经在相机中饱和，压缩只会压低剩余亮度范围，不能恢复真实纹理。

## 评价 API 约定

程序发送 POST multipart/form-data：模型字段是 model.zip（包含 OBJ/PLY、同目录 MTL 和相对路径贴图），metadata 字段是 JSON 字符串。密钥通过配置鉴权头发送，默认 Authorization: Bearer <key>；密钥不会写入配置 JSON、manifest 或日志。响应应为 JSON 对象；界面直接展示 score、metrics、text/evaluation/summary/message 等后端字段，不自行推断评分。

## 已验证

python -m compileall -q glacier_demo/glacier glacier_demo/main.py 通过；虚拟环境已成功安装 PySide6、VTK、OpenCV、Requests Toolbelt 和 psutil，并通过 Qt/VTK/UI 导入检查。未在本机运行真实 GPU VGGT、COLMAP/OpenMVS 或用户 API，因此实际算法效果与 API 字段需在目标环境联调。
