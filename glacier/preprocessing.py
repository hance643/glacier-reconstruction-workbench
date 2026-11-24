"""可复现的冰雪影像处理；预览与实际任务调用同一个函数。

增强仅改变可见亮度/对比度，无法恢复传感器饱和区域，也不会创造真实纹理。
保留原始文件；批处理输出无损 PNG，不做几何变形、不删除图像区域。
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


@dataclass(frozen=True)
class PreprocessConfig:
    highlight: float = 0.15       # 0~0.8，压缩高亮端，避免过度灰化雪面
    clahe: float = 1.5            # 0 表示关闭；局部对比度增强应适度
    gamma: float = 1.0            # <1 提亮，>1 压暗
    denoise: float = 0.0          # 双边滤波强度，默认关闭以保留弱纹理
    sharpen: float = 0.0          # 轻度反锐化，默认关闭以避免伪特征
    max_edge: int = 2400          # 0 表示原尺寸；不裁剪


def collect_images(paths):
    """同时支持拖入文件/文件夹；递归查找，解析绝对路径后去重。"""
    found = set()
    for entry in paths:
        path = Path(entry).expanduser().resolve()
        candidates = path.rglob('*') if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                found.add(candidate.resolve())
    return sorted(found, key=lambda p: str(p).casefold())


def read_rgb(path):
    """Pillow 支持中文路径；统一 EXIF 朝向。当前框架统一转为 8-bit RGB。"""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()


def resize_long_edge(rgb, edge):
    height, width = rgb.shape[:2]
    if edge and max(height, width) > edge:
        scale = edge / max(height, width)
        return cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    return rgb.copy()


def preprocess(rgb, cfg):
    """在 Lab 亮度通道处理以减少偏色，所有图像使用任务启动时的参数快照。"""
    rgb = resize_long_edge(rgb, cfg.max_edge)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    # 单调高光压缩：只压缩亮度超过 0.65 的部分，保持亮暗顺序。
    excess = np.maximum(luminance - 0.65, 0) / 0.35
    luminance -= cfg.highlight * 0.17 * excess**2
    luminance = np.clip(luminance, 0, 1) ** cfg.gamma
    channel = np.uint8(np.round(luminance * 255))
    if cfg.clahe > 0:
        channel = cv2.createCLAHE(clipLimit=cfg.clahe, tileGridSize=(8, 8)).apply(channel)
    if cfg.denoise > 0:
        channel = cv2.bilateralFilter(channel, 5, cfg.denoise, 3)
    if cfg.sharpen > 0:
        smooth = cv2.GaussianBlur(channel, (0, 0), 1.0)
        channel = cv2.addWeighted(channel, 1 + cfg.sharpen, smooth, -cfg.sharpen, 0)
    lab[:, :, 0] = channel
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def write_png(path, rgb):
    """编码后写字节，避免 OpenCV 在部分 Windows 环境下不识别中文路径。"""
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f'图像编码失败：{path}')
    Path(path).write_bytes(encoded.tobytes())
