# 冰川三维重建工作台

一个用于冰川影像重建实验的 Windows 桌面工具。它负责整理输入照片、执行预处理、调用外部重建程序、查看模型并提交模型评价；VGGT、COLMAP 和 OpenMVS 本身需要另行安装和配置。

## 功能

- 导入多张照片，预览并调整高光压缩、局部对比度、Gamma、降噪和锐化。
- 运行 VGGT（路径 A）或 COLMAP + OpenMVS（路径 B），查看任务进度和日志。
- 浏览 OBJ/PLY 模型；VTK 可用于显示网格、纹理和点云颜色。
- 可选调用 DeepSeek Vision 或自建 HTTP 评价服务。
- 在本机记住工具路径、照片列表和最近结果，便于下次恢复。

## 环境要求

- Windows 10/11，Python 3.11 或更高版本。
- 桌面程序依赖见 [`requirements.txt`](requirements.txt)。
- 重建路径还需要独立安装相应的 VGGT、COLMAP 和/或 OpenMVS，以及可用的显卡驱动、CUDA 和模型权重（取决于所选方法）。这些工具不由本项目自动下载或安装。

## 安装与启动

在项目目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

也可以运行 `python launch.py`。检查 Python 依赖是否可导入：

```powershell
.venv\Scripts\python.exe launch.py --check
```

## 配置重建工具

启动后，程序会尝试查找 VGGT、COLMAP 和 OpenMVS。也可以在“重建路径与输出”区域手动选择程序或目录。检测到多个版本时，可从候选列表中选择。

- VGGT Python：选择已安装 VGGT 依赖的 Python 解释器；VGGT 仓库目录应包含项目要求的入口脚本。
- COLMAP：选择 `colmap.exe`，或确保 `colmap` 已加入 `PATH`。
- OpenMVS：选择包含 `InterfaceCOLMAP`、`DensifyPointCloud`、`ReconstructMesh`、`RefineMesh` 和 `TextureMesh` 等程序的 `bin` 目录。

自动检测只确认候选位置，不保证模型权重、显卡环境或全部运行依赖可用。程序开始任务时还会检查工具调用结果；详细信息见界面日志和任务目录中的 `task.log`。

## 基本使用

1. 导入同一场景的至少两张照片。
2. 按需调整预处理参数；原始照片只读取，不会被覆盖。
3. 选择一条或两条重建路径，设置输出目录并启动任务。
4. 在模型页加载或查看结果，在任务日志页检查运行状态。

两条重建路径分别运行；其坐标原点、方向和尺度未必相同，因此并排展示不能直接作为几何配准或精度比较。冰雪高光若已在原始照片中饱和，后处理无法恢复已丢失的纹理信息。

## 结果与本机数据

每次任务在输出目录下创建独立的 `run_时间_UUID` 文件夹，通常包含预处理图像、中间结果、`manifest.json`、`task.log` 和可选的评价结果。具体内容取决于所选方法和运行是否成功。

应用在项目的 `.local/` 中保存本机工具路径和恢复会话信息。这里可能包含 Windows 用户目录、照片位置和模型位置，因此不应上传或提交。清空照片列表后，程序会在后续重建成功时尝试清理旧任务目录；请在清理前确认其中没有需要保留的结果。

## 可选：模型评价 API

评价功能默认不启用。启用并配置服务后：

- DeepSeek Vision 模式会发送模型的斜视、侧视和俯视预览图及几何摘要到所配置的 DeepSeek 服务。
- 自建服务使用 multipart/form-data 上传模型归档和 metadata。
- API 密钥仅在运行期间保存在界面内存中；应用不会将密钥写入项目配置或恢复会话文件。不要把密钥写进源码、README、日志、截图或公开问题报告。

当前内置评价结果采用“90 分展示基础分 + 最多 10 分质量差异”的鼓励性计分方式。这是竞赛演示口径，不是独立精度验证，也不表示不同模型的质量差异只在 10 分以内。没有外部参考数据时，坐标和测绘精度仍属未验证。

## GitHub 上传前检查

仓库的 `.gitignore` 排除了虚拟环境、本机配置、日志、运行结果、密钥文件和常见模型权重。上传前仍应检查待提交文件和 Git 历史：忽略规则不会从既有提交中删除文件，也不会隐藏已经被 Git 跟踪的文件。

```powershell
git status --short
git ls-files
git check-ignore -v .venv .local logs outputs
```

`git ls-files` 中不应出现 `.local/`、`.venv/`、`logs/`、`outputs/`、API 密钥或私人照片。提交记录还包含 Git 作者姓名和邮箱；如不希望公开真实邮箱，请先在 GitHub 开启邮箱隐私，并将 Git 提交邮箱改为 GitHub 提供的 `noreply` 地址。把仓库设为私有可限制仓库访问者，但不能替代密钥管理。

## 项目结构

```text
glacier_demo/
├── assets/             # 应用使用的静态资源
├── glacier/            # 界面、预处理、重建调度、模型查看与评价
├── outputs/            # 本机任务结果（运行时生成，不纳入 Git）
├── logs/               # 本机启动日志（运行时生成，不纳入 Git）
├── main.py             # 桌面应用入口
├── launch.py           # 启动与依赖检查入口
├── requirements.txt    # 桌面版 Python 依赖
└── README.md
```
