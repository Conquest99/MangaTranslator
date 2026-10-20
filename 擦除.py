"""第 4 步：抹字 / 图像修复（复用 BallonsTranslator 的 inpaint 模块）。

前面几步已经把文字「找出来、读出来、翻好」了，但底图上还印着原文。
这一步回答：**把原文从画面里擦掉，且尽量不破坏背景**。

    img + mask + List[TextBlock] ──► inpainted_img

真正干活的不只是那个神经网络，`InpainterBase.inpaint()` 做了四件事：
    1. 按块处理：每块先 expand 1.7 倍窗口，只把小图送模型（省显存、更准）
    2. mask 精修：用 canny/flood-fill 在块内把文字像素抠得更贴边
    3. 纯色气泡走捷径：背景足够单一时直接填中位色，不跑网络
    4. 显存不够时自动退回 CPU（memory_safe_inpaint）

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\擦除.py
    整合\\.venv\\Scripts\\python.exe 整合\\擦除.py --image D:\\manga\\01.jpg
    整合\\.venv\\Scripts\\python.exe 整合\\擦除.py --force-model   # 强制走神经网络
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

INVOKE_CWD = Path.cwd()
os.chdir(BT_ROOT)  # BallonsTranslator 用相对路径找 config/ 和 data/models/
sys.path.insert(0, str(BT_ROOT))

import cv2
import numpy as np

from ballontranslator.modules.inpaint import INPAINTERS
from ballontranslator.utils import config as program_config
from ballontranslator.utils import io_utils
from ballontranslator.utils.textblock import TextBlock

DEFAULT_IMAGE = ROOT / 'samples' / 'sample_page.png'
OUT_DIR = ROOT / 'outputs'


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (INVOKE_CWD / path).resolve()


def find_blocks_json(stem: str) -> Path:
    """优先用翻译结果（信息最全），没有就退回 OCR 结果。"""
    for suffix in ('_trans.json', '_ocr.json'):
        candidate = OUT_DIR / f'{stem}{suffix}'
        if candidate.exists():
            return candidate
    raise SystemExit(
        f'找不到 {stem} 的文字块数据，先跑第 2 步：整合\\引入ORC.py'
    )


def load_blocks(blocks_json: Path) -> list:
    records = json.loads(blocks_json.read_text(encoding='utf-8'))
    blocks = []
    for r in records:
        text = r.get('text_lines') or [r.get('source', r.get('text', ''))]
        blk = TextBlock(
            xyxy=r['xyxy'],
            lines=[np.asarray(line, dtype=np.float32) for line in r['lines']],
            angle=r.get('angle', 0),
            src_is_vertical=r.get('src_is_vertical', False),
            text=list(text),
        )
        blk._detected_font_size = r.get('font_size', -1)
        blocks.append(blk)
    return blocks


def build_inpainter(key: str, device: str, inpaint_size: int):
    entry = INPAINTERS.get(key)
    if entry is None:
        available = ', '.join(INPAINTERS.module_dict.keys())
        raise SystemExit(f'没有名为 {key!r} 的修复器。可用：{available}')

    inpainter_cls = entry.resolve() if hasattr(entry, 'resolve') else entry
    params = program_config.pcfg.module.get_params('inpainter').get(key) or {}
    inpainter = inpainter_cls(**params)
    if device != 'auto':
        inpainter.set_param_value('device', device)
    if inpaint_size is not None:
        inpainter.set_param_value('inpaint_size', inpaint_size)
    return inpainter


def main():
    parser = argparse.ArgumentParser(description='第 4 步：抹字 / 图像修复')
    parser.add_argument('--image', default=None, help='漫画页路径，默认用 samples/sample_page.png')
    parser.add_argument('--blocks', default=None, help='文字块 json，默认自动找 *_trans.json / *_ocr.json')
    parser.add_argument('--mask', default=None, help='文字 mask，默认 outputs/<stem>_mask.png')
    parser.add_argument('--inpainter', default='lama_large_512px', help='注册表 key，如 lama_large_512px / aot / patchmatch')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--inpaint-size', type=int, default=1024, help='推理尺寸：512/768/1024/1536/2048')
    parser.add_argument('--force-model', action='store_true',
                        help='关掉“纯色气泡直接填色”的快捷路径，强制走神经网络')
    args = parser.parse_args()

    image_path = resolve_path(args.image) if args.image else DEFAULT_IMAGE
    if not image_path.exists():
        raise SystemExit(f'图片不存在：{image_path}')
    stem = image_path.stem

    blocks_json = resolve_path(args.blocks) if args.blocks else find_blocks_json(stem)
    mask_path = resolve_path(args.mask) if args.mask else OUT_DIR / f'{stem}_mask.png'
    if not mask_path.exists():
        raise SystemExit(f'找不到 mask：{mask_path}\n先跑第 1 步：整合\\引入引擎.py')

    program_config.load_config()
    if args.force_model:
        program_config.pcfg.module.check_need_inpaint = False

    img = io_utils.imread(str(image_path))
    mask = io_utils.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    blocks = load_blocks(blocks_json)

    print(f'图片 {image_path.name} {img.shape[1]}x{img.shape[0]} | 文字块 {len(blocks)} '
          f'| mask 覆盖率 {100.0 * (mask > 127).mean():.2f}%')

    inpainter = build_inpainter(args.inpainter, args.device, args.inpaint_size)
    print(f'修复器 {inpainter.name} | device={inpainter.get_param_value("device")} | '
          f'inpaint_size={inpainter.get_param_value("inpaint_size")} | '
          f'force_model={args.force_model}')

    t0 = time.time()
    # inpaint() 会改 mask（逐块处理时把已处理的区域置零），传副本避免影响原 mask
    inpainted = inpainter.inpaint(img, mask.copy(), blocks)
    cost = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{stem}_inpainted.png'
    cmp_path = OUT_DIR / f'{stem}_inpaint_compare.png'
    io_utils.imwrite(str(out_path), inpainted)
    io_utils.imwrite(str(cmp_path), np.hstack([img, inpainted]))

    print(f'\n抹字耗时 {cost:.2f}s')
    print(f'已保存：\n  结果 -> {out_path}\n  对比 -> {cmp_path}  （左：原图，右：抹字后）')


if __name__ == '__main__':
    main()
