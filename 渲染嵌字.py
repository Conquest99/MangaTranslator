"""第 5 步：嵌字 / 排版渲染（不依赖 BallonsTranslator）。

前四步产出了「空白底图 + 每块译文 + 几何信息」，最后一步把译文画回去。
这也是整条链路里唯一**没有现成模块可调用**的一步——BallonsTranslator 用 Qt 的
QGraphicsTextItem + 自定义 layout 实现，这里用 PIL 做一个最小可用版，逻辑一一对应：

    译文 + xyxy + 竖排标记 ──► 选字体 ──► 试字号（放不下就缩小）──► 逐字/逐行绘制

关键规则：
    1. 竖排：从上到下，列从右往左（日漫习惯）
    2. 字号：从检测到的字号开始，放不下就逐号递减，直到塞进气泡
    3. 描边：白色细描边，保证压在网点/背景上也看得清
    4. 颜色：优先继承 OCR 估出的原文前景色

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\渲染嵌字.py
    整合\\.venv\\Scripts\\python.exe 整合\\渲染嵌字.py --image 整合\\data\\p1.jpg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / 'outputs'
DEFAULT_IMAGE = Path(__file__).resolve().parent / 'samples' / 'sample_page.png'
DEFAULT_FONT = r'C:\Windows\Fonts\msyh.ttc'

CLOSING_PUNCT = set('。、，．,.！？!?）)」』】》〉…—～ー')
OPENING_PUNCT = set('（(「『【《〈')


def load_records(stem: str, trans_path: Path):
    """译文 + 几何来自 *_trans.json，前景色来自 *_ocr.json（按 id 对齐）。"""
    records = json.loads(trans_path.read_text(encoding='utf-8'))
    colors = {}
    ocr_path = OUT_DIR / f'{stem}_ocr.json'
    if ocr_path.exists():
        colors = {
            r['id']: tuple(r.get('fg_color', (0, 0, 0)))
            for r in json.loads(ocr_path.read_text(encoding='utf-8'))
        }
    return records, colors


def text_color(rgb: tuple) -> tuple:
    """背景色被误当成前景色时（接近纯白），退回黑色。"""
    if sum(rgb) / 3 > 200:
        return (0, 0, 0)
    return tuple(int(c) for c in rgb)


def split_columns(text: str, n_cols: int, per_col: int) -> list:
    """按列数均分，并做最基本的行首/行尾禁则（避免标点单独占一列）。"""
    n = len(text)
    base = math.ceil(n / n_cols)
    lengths = [base] * (n_cols - 1) + [n - base * (n_cols - 1)]

    for k in range(1, n_cols):
        start = sum(lengths[:k])
        if start >= n:
            break
        # 行首不能是收尾标点 → 把上一列最后一个字挪下来
        if text[start] in CLOSING_PUNCT and lengths[k - 1] > 1 and lengths[k] + 1 <= per_col:
            lengths[k - 1] -= 1
            lengths[k] += 1
        # 行尾不能是起始标点 → 把它挪到下一列
        end = sum(lengths[:k])
        if text[end - 1] in OPENING_PUNCT and lengths[k - 1] > 1 and lengths[k] + 1 <= per_col:
            lengths[k - 1] -= 1
            lengths[k] += 1

    cols, i = [], 0
    for length in lengths:
        cols.append(text[i:i + length])
        i += length
    return [c for c in cols if c]


def fit_vertical(text: str, box_w: int, box_h: int, start_size: float, font_path: str, min_size: int = 10):
    """竖排：一列最多放几个字 → 需要几列 → 总宽度够不够，不够就缩字号。"""
    size = max(min_size, int(round(start_size)))
    while size >= min_size:
        step = size * 1.06
        per_col = max(1, int(box_h // step))
        n_cols = math.ceil(len(text) / per_col)
        cols = split_columns(text, n_cols, per_col)
        if max(len(c) for c in cols) * step <= box_h + 1 and len(cols) * size * 1.12 <= box_w + 1:
            return ImageFont.truetype(font_path, size), cols
        size -= 1
    font = ImageFont.truetype(font_path, min_size)
    per_col = max(1, int(box_h // (min_size * 1.06)))
    return font, split_columns(text, math.ceil(len(text) / per_col), per_col)


def wrap_horizontal(text: str, draw: ImageDraw.ImageDraw, font, max_w: int):
    lines, cur = [], ''
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_w or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    # 避免行首出现收尾标点
    for i in range(1, len(lines)):
        while lines[i] and lines[i][0] in CLOSING_PUNCT and len(lines[i - 1]) > 1:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    return lines


def fit_horizontal(text: str, draw, box_w: int, box_h: int, start_size: float, font_path: str, min_size: int = 10):
    size = max(min_size, int(round(start_size)))
    while size >= min_size:
        font = ImageFont.truetype(font_path, size)
        lines = wrap_horizontal(text, draw, font, box_w)
        if len(lines) * size * 1.25 <= box_h:
            return font, lines
        size -= 1
    font = ImageFont.truetype(font_path, min_size)
    return font, wrap_horizontal(text, draw, font, box_w)


def draw_vertical(draw, cols, box, font, fill, stroke_w):
    x1, y1, x2, y2 = box
    size = font.size
    step = size * 1.06
    col_w = size * 1.12
    max_len = max(len(c) for c in cols)
    total_w = len(cols) * col_w
    total_h = max_len * step
    x_right = (x1 + x2) / 2 + total_w / 2 - col_w / 2
    top = (y1 + y2) / 2 - total_h / 2 + step / 2
    for ci, col in enumerate(cols):
        x = x_right - ci * col_w  # 列从右往左
        for ri, ch in enumerate(col):
            draw.text(
                (x, top + ri * step), ch,
                font=font, anchor='mm', fill=fill,
                stroke_width=stroke_w, stroke_fill=(255, 255, 255),
            )


def draw_horizontal(draw, lines, box, font, fill, stroke_w):
    x1, y1, x2, y2 = box
    line_h = font.size * 1.25
    cx = (x1 + x2) / 2
    top = (y1 + y2) / 2 - len(lines) * line_h / 2 + line_h / 2
    for j, line in enumerate(lines):
        draw.text(
            (cx, top + j * line_h), line,
            font=font, anchor='mm', fill=fill,
            stroke_width=stroke_w, stroke_fill=(255, 255, 255),
        )


def main():
    parser = argparse.ArgumentParser(description='第 5 步：嵌字 / 排版渲染')
    parser.add_argument('--image', default=None, help='原图（用来推导文件名），默认 samples/sample_page.png')
    parser.add_argument('--trans', default=None, help='译文 json，默认 outputs/<stem>_trans.json')
    parser.add_argument('--base', default=None, help='底图，默认 outputs/<stem>_inpainted.png')
    parser.add_argument('--font', default=DEFAULT_FONT, help='字体文件，默认微软雅黑')
    parser.add_argument('--min-size', type=int, default=10, help='字号下限')
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else DEFAULT_IMAGE
    stem = image_path.stem

    trans_path = Path(args.trans) if args.trans else OUT_DIR / f'{stem}_trans.json'
    base_path = Path(args.base) if args.base else OUT_DIR / f'{stem}_inpainted.png'
    for p in (trans_path, base_path):
        if not p.exists():
            raise SystemExit(f'找不到 {p}\n先跑完前面几步：检测 -> OCR -> 翻译 -> 抹字')

    records, colors = load_records(stem, trans_path)
    canvas = Image.open(base_path).convert('RGB')
    draw = ImageDraw.Draw(canvas)

    rendered = 0
    for r in records:
        text = (r.get('translation') or '').strip()
        if not text:
            continue
        box = r['xyxy']
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        if box_w <= 2 or box_h <= 2:
            continue

        start_size = max(args.min_size, float(r.get('font_size', 24)))
        fill = text_color(colors.get(r['id'], (0, 0, 0)))
        stroke_w = max(1, round(start_size / 16))

        if r.get('src_is_vertical', False):
            font, cols = fit_vertical(text, box_w, box_h, start_size, args.font, args.min_size)
            stroke_w = max(1, round(font.size / 16))
            draw_vertical(draw, cols, box, font, fill, stroke_w)
            print(f"  #{r['id']} 竖排 {font.size}px ({len(cols)} 列)  {text}")
        else:
            font, lines = fit_horizontal(text, draw, box_w, box_h, start_size, args.font, args.min_size)
            stroke_w = max(1, round(font.size / 16))
            draw_horizontal(draw, lines, box, font, fill, stroke_w)
            print(f"  #{r['id']} 横排 {font.size}px ({len(lines)} 行)  {text}")
        rendered += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{stem}_rendered.png'
    compare_path = OUT_DIR / f'{stem}_render_compare.png'
    canvas.save(out_path)
    Image.new('RGB', (canvas.width * 2, canvas.height), 'white').save(compare_path)
    side = Image.open(compare_path)
    side.paste(Image.open(base_path).convert('RGB'), (0, 0))
    side.paste(canvas, (canvas.width, 0))
    side.save(compare_path)

    print(f'\n共渲染 {rendered} 块')
    print(f'已保存：\n  成品 -> {out_path}\n  对比 -> {compare_path}  （左：抹字底图，右：嵌字成品）')


if __name__ == '__main__':
    main()
