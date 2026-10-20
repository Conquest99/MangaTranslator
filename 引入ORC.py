"""第 2 步：OCR（复用 BallonsTranslator 的 mit48px 模型）。

第一步给了我们「哪里有字」，这一步回答「写的是什么」：

    img + List[TextBlock] ──► 每块 blk.text = ['第一行', '第二行', ...]
                             同时给出 blk.fg_colors / blg.bg_colors（原文前景/背景色）

注意 OCR 是**逐块**跑的，不是整图跑一次：
    - 按 blk.lines 的多边形把每行文字抠出来、转成水平（竖排会被转 90°）
    - 缩放到固定高度 48px 后送进模型
    - 结果写回同一个 TextBlock，后面的翻译/抹字/嵌字继续用它

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\引入ORC.py
    整合\\.venv\\Scripts\\python.exe 整合\\引入ORC.py --image D:\\manga\\01.jpg
    整合\\.venv\\Scripts\\python.exe 整合\\引入ORC.py --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BT_ROOT = (ROOT.parent / 'BallonsTranslator').resolve()
if not (BT_ROOT / 'ballontranslator').is_dir():
    raise SystemExit(f'找不到 BallonsTranslator：{BT_ROOT}')

# 记住用户执行命令时的工作目录：下面会 chdir，相对路径参数要靠它还原
INVOKE_CWD = Path.cwd()

os.chdir(BT_ROOT)  # BallonsTranslator 用相对路径找 config/ 和 data/models/
sys.path.insert(0, str(BT_ROOT))

import numpy as np

from ballontranslator.modules.ocr import OCR
from ballontranslator.utils import config as program_config
from ballontranslator.utils import io_utils
from ballontranslator.utils.textblock import TextBlock
DEFAULT_IMAGE = ROOT / 'samples' / 'sample_page.png'
OUT_DIR = ROOT / 'outputs'
MODEL_CKPT = BT_ROOT / 'data' / 'models' / 'ocr_ar_48px.ckpt'
MODEL_DICT = BT_ROOT / 'data' / 'alphabet-all-v7.txt'


def load_stage1():
    """按文件路径加载第 1 步的脚本，复用它的检测器。

    直接用 import 名字也行，但从文件加载更稳（不依赖中文模块名）。
    """
    spec = importlib.util.spec_from_file_location('stage1_detector', ROOT / '引入引擎.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ocr(engine: str, device: str):
    entry = OCR.get(engine)
    if entry is None:
        available = ', '.join(OCR.module_dict.keys())
        raise SystemExit(f'没有名为 {engine!r} 的 OCR。可用：{available}')
    # 和检测器一样：注册表里是 ModuleSpec，resolve() 才真正 import 出类
    ocr_cls = entry.resolve() if hasattr(entry, 'resolve') else entry
    # 配置里保存过的参数（device / chunk_size / rec_batch_num 等）
    params = program_config.pcfg.module.get_params('ocr').get(engine) or {}
    ocr = ocr_cls(**params)
    if device != 'auto':
        ocr.set_param_value('device', device)
    return ocr


def load_blocks(blocks_path: Path) -> list:
    """从第 1 步保存的 json 还原 TextBlock（展示“阶段之间只传 TextBlock”）。"""
    records = json.loads(blocks_path.read_text(encoding='utf-8'))
    blocks = []
    for r in records:
        blk = TextBlock(
            xyxy=r['xyxy'],
            lines=[np.asarray(line, dtype=np.float32) for line in r['lines']],
            angle=r['angle'],
            src_is_vertical=r['src_is_vertical'],
        )
        blk._detected_font_size = r['detected_font_size']
        blocks.append(blk)
    return blocks


def main():
    parser = argparse.ArgumentParser(description='第 2 步：OCR')
    parser.add_argument('--image', default=None, help='漫画页路径，默认用 samples/sample_page.png')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--detect-size', type=int, default=1280, help='文本检测输入尺寸')
    parser.add_argument('--blocks', default=None, help='复用第 1 步的 *_blocks.json，跳过重新检测')
    parser.add_argument('--ocr-engine', default='mit48px',
                        help='OCR 引擎：mit48px / manga_ocr / PaddleOCRVLManga / ppv6_onnx')
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else DEFAULT_IMAGE
    if not image_path.is_absolute():
        image_path = (INVOKE_CWD / image_path).resolve()
    if not image_path.exists():
        raise SystemExit(f'图片不存在：{image_path}')
    if args.ocr_engine == 'mit48px' and (not MODEL_CKPT.exists() or not MODEL_DICT.exists()):
        raise SystemExit(
            f'缺少 OCR 模型：\n  {MODEL_CKPT}\n  {MODEL_DICT}\n'
            '下载 ocr48px.zip 后解压到上述位置：\n'
            '  https://hf-mirror.com/dreMaz/mit_models/resolve/main/ocr48px.zip'
        )

    program_config.load_config()
    img = io_utils.imread(str(image_path))
    print(f'输入图片 {image_path.name}  尺寸={img.shape[1]}x{img.shape[0]}')

    # ---------- ① 拿到文字块：复用 json，或者重新检测 ----------
    if args.blocks:
        blocks_path = Path(args.blocks)
        if not blocks_path.is_absolute():
            blocks_path = (INVOKE_CWD / blocks_path).resolve()
        blocks = load_blocks(blocks_path)
        print(f'从 {blocks_path.name} 载入 {len(blocks)} 个文字块')
    else:
        stage1 = load_stage1()
        detector = stage1.build_detector(args.device, args.detect_size)
        t0 = time.time()
        _, blocks = detector.detect(img)
        print(f'文本检测 {time.time() - t0:.2f}s，得到 {len(blocks)} 个文字块')

    if not blocks:
        raise SystemExit('没有检测到文字块，OCR 无从下手')

    # ---------- ② 逐块 OCR ----------
    ocr = build_ocr(args.ocr_engine, args.device)
    print(f'OCR {ocr.name} | device={ocr.get_param_value("device")}')

    t0 = time.time()
    result = ocr.run_ocr(img, blocks)  # 第一次调用会在这里加载模型
    if isinstance(result, list):
        blocks = result
    cost = time.time() - t0

    # ---------- ③ 保存结果 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    json_path = OUT_DIR / f'{stem}_ocr.json'
    txt_path = OUT_DIR / f'{stem}_ocr.txt'
    crop_dir = OUT_DIR / f'{stem}_crops'
    crop_dir.mkdir(exist_ok=True)

    records = []
    lines_out = []
    for i, blk in enumerate(blocks):
        fg, bg = blk.get_font_colors()
        text = blk.get_text()
        records.append({
            'id': i,
            'xyxy': [int(v) for v in blk.xyxy],
            'angle': int(blk.angle),
            'src_is_vertical': bool(blk.src_is_vertical),
            'font_size': round(float(blk._detected_font_size), 2),
            'text': text,
            'text_lines': list(blk.text) if not isinstance(blk.text, str) else [blk.text],
            'fg_color': [int(v) for v in fg],
            'bg_color': [int(v) for v in bg],
            'lines': [np.asarray(line).round(1).tolist() for line in blk.lines],
        })
        lines_out.append(f'#{i}\t{text}')

        # 把模型真正“看到”的那块图存下来（竖排已转成水平）
        for j in range(len(blk.lines)):
            region = blk.get_transformed_region(img, j, 48)
            io_utils.imwrite(str(crop_dir / f'blk{i}_line{j}.png'), region)

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    txt_path.write_text('\n'.join(lines_out) + '\n', encoding='utf-8')

    print(f'\nOCR 耗时 {cost:.2f}s：')
    for r in records:
        print(f"  #{r['id']} 竖排={r['src_is_vertical']} 字号={r['font_size']} "
              f"前景={tuple(r['fg_color'])} 背景={tuple(r['bg_color'])}")
        print(f"      {r['text']}")

    print(f'\n已保存：\n  文本 -> {txt_path}\n  数据 -> {json_path}\n  切图 -> {crop_dir}')


if __name__ == '__main__':
    main()
