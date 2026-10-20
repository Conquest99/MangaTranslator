"""第 1 步：文本检测（复用 BallonsTranslator 的 ctd 模型）。

这一步只回答一个问题：**文字在哪、大概多大**。

    图片 ──► mask（文字像素=255） + List[TextBlock]
                              └─ xyxy 外接框 / lines 行多边形 / angle 倾斜角
                                 src_is_vertical 是否竖排 / _detected_font_size 估计字号

它不涉及 OCR、翻译、抹字，后面每一步都从这份 TextBlock 列表继续。

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\引入引擎.py
    整合\\.venv\\Scripts\\python.exe 整合\\引入引擎.py --image D:\\manga\\01.jpg
    整合\\.venv\\Scripts\\python.exe 整合\\引入引擎.py --device cpu --detect-size 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BT_ROOT = (ROOT.parent / 'BallonsTranslator').resolve()
if not (BT_ROOT / 'ballontranslator').is_dir():
    raise SystemExit(f'找不到 BallonsTranslator：{BT_ROOT}')

# 记住用户执行命令时的工作目录：下面会 chdir，相对路径 --image 要靠它还原
INVOKE_CWD = Path.cwd()

# BallonsTranslator 内部用相对路径找 config/ 和 data/models/，所以先切到它的根目录
os.chdir(BT_ROOT)
sys.path.insert(0, str(BT_ROOT))

import cv2
import numpy as np

from ballontranslator.modules.textdetector import TEXTDETECTORS
from ballontranslator.utils import config as program_config
from ballontranslator.utils import io_utils

DEFAULT_IMAGE = ROOT / 'samples' / 'sample_page.png'
OUT_DIR = ROOT / 'outputs'
MODEL_PT = BT_ROOT / 'data' / 'models' / 'comictextdetector.pt'
MODEL_ONNX = BT_ROOT / 'data' / 'models' / 'comictextdetector.pt.onnx'


def build_detector(device: str, detect_size: int):
    """按注册表里的 key 取模块，再用参数覆盖默认值。"""
    entry = TEXTDETECTORS.get('ctd')
    # 模块是懒加载的：拿到的是 ModuleSpec，resolve() 才真正 import 出类
    detector_cls = entry.resolve() if hasattr(entry, 'resolve') else entry
    detector = detector_cls()

    if device != 'auto':
        detector.set_param_value('device', device)
    detector.set_param_value('detect_size', detect_size)

    if detector.get_param_value('device') == 'cpu' and not MODEL_ONNX.exists():
        raise SystemExit(
            'CPU 模式走的是 ONNX 模型，当前只有 PyTorch 权重：\n'
            f'  {MODEL_PT}\n'
            '要么改用 --device auto 走 GPU，要么下载 comictextdetector.pt.onnx'
        )
    return detector


def draw_blocks(img: np.ndarray, blocks: list) -> np.ndarray:
    """把检测结果画在原图上，方便肉眼检查。"""
    canvas = img.copy()
    for i, blk in enumerate(blocks):
        for line in blk.lines:
            pts = np.asarray(line, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)
        x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
        cv2.putText(canvas, str(i), (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description='第 1 步：文本检测')
    parser.add_argument('--image', default=None, help='漫画页路径，默认用 samples/sample_page.png')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--detect-size', type=int, default=1280, help='检测输入尺寸，越大越准也越慢')
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else DEFAULT_IMAGE
    if not image_path.is_absolute():
        image_path = (INVOKE_CWD / image_path).resolve()
    if not image_path.exists():
        raise SystemExit(f'图片不存在：{image_path}')
    if not MODEL_PT.exists():
        raise SystemExit(
            f'缺少检测模型：{MODEL_PT}\n'
            '从镜像下载：\n'
            '  curl -L -o data/models/comictextdetector.pt '
            'https://hf-mirror.com/dreMaz/mit_models/resolve/main/comictextdetector.pt'
        )

    program_config.load_config()
    detector = build_detector(args.device, args.detect_size)
    print(f'检测器 ctd | device={detector.get_param_value("device")} | '
          f'detect_size={detector.get_param_value("detect_size")}')

    img = io_utils.imread(str(image_path))
    print(f'输入图片 {image_path.name}  尺寸={img.shape[1]}x{img.shape[0]}')

    t0 = time.time()
    mask, blocks = detector.detect(img)  # 第一次调用会在这里加载模型
    cost = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = OUT_DIR / f'{stem}_mask.png'
    vis_path = OUT_DIR / f'{stem}_detect.png'
    json_path = OUT_DIR / f'{stem}_blocks.json'

    io_utils.imwrite(str(mask_path), mask)
    io_utils.imwrite(str(vis_path), draw_blocks(img, blocks))

    records = []
    for i, blk in enumerate(blocks):
        records.append({
            'id': i,
            'xyxy': [int(v) for v in blk.xyxy],
            'angle': int(blk.angle),
            'src_is_vertical': bool(blk.src_is_vertical),
            'detected_font_size': round(float(blk._detected_font_size), 2),
            'lines': [np.asarray(line).round(1).tolist() for line in blk.lines],
        })
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n检测耗时 {cost:.2f}s，找到 {len(blocks)} 个文字块：')
    for r in records:
        print(f"  #{r['id']} xyxy={r['xyxy']} 竖排={r['src_is_vertical']} "
              f"倾斜={r['angle']}° 估计字号={r['detected_font_size']} 行数={len(r['lines'])}")

    print(f'\n已保存：\n  mask   -> {mask_path}\n  可视化 -> {vis_path}\n  数据   -> {json_path}')


if __name__ == '__main__':
    main()
