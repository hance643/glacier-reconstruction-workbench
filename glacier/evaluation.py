"""HTTP 评价适配器：不在客户端计算/编造分数，仅展示后端 JSON。

请求：multipart/form-data，model=ZIP 文件，metadata=JSON 字符串。
ZIP 包含模型、同目录及子目录中的材质/纹理。接口字段名、鉴权头、
响应根路径均可配置；密钥只保存在内存，不进入任务清单或日志。
"""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
import zipfile

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor


@dataclass(frozen=True)
class APIConfig:
    enabled: bool = False
    url: str = ''
    key: str = ''
    auth_header: str = 'Authorization'
    auth_prefix: str = 'Bearer'
    file_field: str = 'model'
    metadata_field: str = 'metadata'
    result_path: str = ''          # 如 data.result；留空使用响应顶层
    timeout: int = 120            # 读取超时；连接超时单独限制为 10 秒


class EvaluationError(RuntimeError):
    pass


def validate_api(cfg):
    parts = urlsplit(cfg.url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise EvaluationError('API 地址必须为不含用户名/密码的完整 HTTP(S) URL。')
    if not cfg.file_field.strip() or not cfg.metadata_field.strip() or cfg.file_field == cfg.metadata_field:
        raise EvaluationError('文件字段与元数据字段必须非空且不同。')
    if any(c in cfg.key + cfg.auth_header + cfg.auth_prefix for c in '\r\n'):
        raise EvaluationError('鉴权配置不能包含换行。')
    if cfg.key and not cfg.auth_header.strip():
        raise EvaluationError('填写密钥时也需要指定鉴权头名称。')


def bundle_model(model, destination, check_cancel=lambda: None):
    """保留纹理相对目录；不跟随指向结果目录以外的符号链接。

外部模型的 MTL/贴图请放在模型同目录或子目录。这里不上传源照片，
不上传数据库、检查点或 .mvs 中间文件。限制总大小避免无限制打包。
"""
    model = Path(model).resolve()
    root = model.parent
    assets = {'.mtl', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    files = [model] + sorted(p for p in root.rglob('*')
                            if p.is_file() and p.suffix.lower() in assets and p != model)
    total = 0
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            check_cancel()
            if not path.resolve().is_relative_to(root):
                raise EvaluationError('纹理链接指向模型目录以外，请先整理为独立模型目录。')
            total += path.stat().st_size
            if total > 2 * 1024**3:
                raise EvaluationError('模型及纹理超过 2 GiB，请简化模型后评价。')
            # 分块写入，让大型模型打包阶段也能及时响应取消。
            with path.open('rb') as src, archive.open(path.relative_to(root).as_posix(), 'w', force_zip64=True) as dst:
                while chunk := src.read(1024 * 1024):
                    check_cancel()
                    dst.write(chunk)


def evaluate(model, metadata, cfg, check_cancel=lambda: None):
    validate_api(cfg)
    check_cancel()
    with tempfile.TemporaryDirectory(prefix='glacier-eval-') as temp:
        archive_path = Path(temp) / 'model.zip'
        bundle_model(model, archive_path, check_cancel)
        headers = {}
        if cfg.key:
            headers[cfg.auth_header] = (cfg.auth_prefix.strip() + ' ' + cfg.key).strip()
        with archive_path.open('rb') as stream:
            # 普通 requests(files=...) 会先把文件读进内存；编码器采用流式上传。
            encoder = MultipartEncoder(fields={
                cfg.file_field: ('model.zip', stream, 'application/zip'),
                cfg.metadata_field: json.dumps(metadata, ensure_ascii=False),
            })
            monitor = MultipartEncoderMonitor(encoder, lambda _: check_cancel())
            headers['Content-Type'] = monitor.content_type
            try:
                # 禁止自动重定向，防止自定义鉴权头被转发给其他服务器；TLS 校验保持开启。
                with requests.post(cfg.url, data=monitor, headers=headers, timeout=(10, cfg.timeout),
                                   allow_redirects=False, stream=True) as response:
                    if not 200 <= response.status_code < 300:
                        raise EvaluationError(f'API 返回 HTTP {response.status_code}，请检查接口及鉴权配置。')
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        check_cancel()
                        data.extend(chunk)
                        if len(data) > 4 * 1024**2:
                            raise EvaluationError('API 响应超过 4 MiB，请返回精简 JSON 指标。')
                    try:
                        payload = json.loads(data)
                    except (ValueError, UnicodeError) as exc:
                        raise EvaluationError('API 未返回有效 JSON。') from exc
            except requests.RequestException as exc:
                # 不直接显示异常原文：某些异常可能含 URL 查询参数或鉴权信息。
                raise EvaluationError(f'HTTP 请求失败（{type(exc).__name__}），请检查网络、证书与超时设置。') from exc
    check_cancel()
    result = payload
    try:
        for key in filter(None, cfg.result_path.split('.')):
            result = result[key]
    except (KeyError, TypeError) as exc:
        raise EvaluationError('响应根路径不存在，请检查 result_path。') from exc
    if not isinstance(result, dict):
        raise EvaluationError('评价结果应是 JSON 对象，包含 score、metrics、text 等后端字段。')
    return {'result': result, 'raw': payload}
