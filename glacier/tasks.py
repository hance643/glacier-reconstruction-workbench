"""面向 UI 的任务编排。每次执行使用不可变参数快照和唯一任务目录。"""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import uuid

from .evaluation import evaluate, validate_api
from .pipeline import Cancelled, validate_tools, run_vggt, run_colmap_openmvs
from .preprocessing import preprocess, read_rgb, write_png


def save_json(path, data):
    """先写临时文件再替换，避免中断时清单只写了一半。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def reconstruction_task(paths, output, methods, prep, tools, api):
    # 拷贝集合，用户在 UI 中继续操作时不会改变已提交任务。
    paths, methods = tuple(paths), tuple(methods)

    def run(report, check):
        if len(paths) < 2:
            raise ValueError('允许单张导入及预览，但多视图重建至少需要两张有重叠的照片。')
        if api.enabled:
            validate_api(api)
        root = Path(output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / (datetime.now().strftime('run_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
        run_dir.mkdir()
        images = run_dir / 'images'
        images.mkdir()
        manifest = {'created_at': datetime.now().astimezone().isoformat(), 'status': 'running',
                    'input_images': [], 'preprocessing': asdict(prep), 'methods': list(methods),
                    'tools': asdict(tools), 'results': {}, 'evaluation_enabled': api.enabled}
        manifest_path = run_dir / 'manifest.json'
        save_json(manifest_path, manifest)
        report({'type': 'run', 'path': str(run_dir)})
        try:
            with (run_dir / 'task.log').open('w', encoding='utf-8', buffering=1) as log:
                def emit(message):
                    log.write(message + '\n')
                    report({'type': 'log', 'text': message})

                # 单条路径配置不正确，不阻断另一条可用路径。
                valid = []
                for method in methods:
                    check()
                    try:
                        validate_tools([method], tools)
                        valid.append(method)
                    except Exception as exc:
                        manifest['results'][method] = {'status': 'failed', 'error': str(exc)}
                        emit(f'路径 {method} 配置检查失败：{exc}')
                if not valid:
                    raise RuntimeError('所选路径均未通过工具配置检查，请查看日志。')
                for index, path in enumerate(paths):
                    check()
                    # 连续唯一名称消除不同子目录同名照片冲突，并固定输入顺序。
                    target = images / f'{index:06d}.png'
                    rgb = preprocess(read_rgb(path), prep)
                    check()
                    write_png(target, rgb)
                    manifest['input_images'].append({'source': str(path), 'processed': target.name,
                                                     'width': rgb.shape[1], 'height': rgb.shape[0]})
                    report({'type': 'progress', 'value': int(20 * (index + 1) / len(paths)),
                            'text': f'图像预处理 {index + 1}/{len(paths)}'})
                save_json(manifest_path, manifest)
                for position, method in enumerate(valid):
                    check()
                    def stage(done, total, text):
                        # 百分比是已完成阶段权重，不是 GPU 迭代进度或剩余时间预测。
                        value = 20 + int(75 * (position + done / total) / len(valid))
                        report({'type': 'progress', 'value': value, 'text': f'路径 {method} · {text}'})
                    try:
                        runner = run_vggt if method == 'A' else run_colmap_openmvs
                        model = runner(images, run_dir / method, tools, emit, stage, check)
                        record = {'status': 'success', 'model': str(model), 'evaluation_status': 'disabled'}
                        manifest['results'][method] = record
                        save_json(manifest_path, manifest)
                        report({'type': 'model', 'method': method, 'path': str(model)})
                        if api.enabled:
                            emit(f'路径 {method}：上传模型并等待 API 评价。')
                            report({'type': 'progress', 'value': None, 'text': f'路径 {method} · API 评价中'})
                            try:
                                result = evaluate(model, {'method': method, 'run_id': run_dir.name,
                                                         'image_count': len(paths), 'preprocessing': asdict(prep)}, api, check)
                                save_json(run_dir / method / 'evaluation.json', result)
                                record['evaluation_status'] = 'success'
                                report({'type': 'evaluation', 'method': method, 'data': result})
                            except Cancelled:
                                record['evaluation_status'] = 'cancelled'
                                raise
                            except Exception as exc:
                                # API 故障不影响已经完成的三维模型，也不阻断另一条路径。
                                record.update(evaluation_status='failed', evaluation_error=str(exc))
                                emit(f'路径 {method} 模型已保存，评价失败：{exc}')
                                report({'type': 'evaluation_error', 'method': method, 'text': str(exc)})
                    except Cancelled:
                        raise
                    except Exception as exc:
                        manifest['results'][method] = {'status': 'failed', 'error': str(exc)}
                        emit(f'路径 {method} 失败：{exc}')
                    save_json(manifest_path, manifest)
                succeeded = sum(r['status'] == 'success' for r in manifest['results'].values())
                manifest['status'] = 'success' if succeeded == len(methods) else ('partial' if succeeded else 'failed')
                report({'type': 'progress', 'value': 100, 'text': f'任务结束：{succeeded}/{len(methods)} 条路径成功'})
        except Cancelled:
            manifest['status'] = 'cancelled'
            raise
        except Exception as exc:
            manifest.update(status='failed', error=str(exc))
            raise
        finally:
            save_json(manifest_path, manifest)
        return {'run_dir': str(run_dir), 'manifest': manifest}
    return run
