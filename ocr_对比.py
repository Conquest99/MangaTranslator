"""在同一页上跑多个 OCR，并用 CER（字符错误率）打分。

打分只针对**人工确认过**的文字块（samples/p1_gt.json），
手写体/拟声词那几块没有可靠 ground truth，只展示结果、不参与评分。

    CER = 编辑距离(预测, 标注) / len(标注)
    CER 越低越好；NFKC 归一化后忽略空格差异。

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\ocr_对比.py
    整合\\.venv\\Scripts\\python.exe 整合\\ocr_对比.py --engines mit48px manga_ocr
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BT_ROOT = (ROOT.parent / 'BallonsTranslator').resolve()
INVOKE_CWD = Path.cwd()
# 不让 transformers 去 import TensorFlow（省启动时间、少一堆日志）
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.chdir(BT_ROOT)  # BallonsTranslator 用相对路径找 config/ 和 data/models/
sys.path.insert(0, str(BT_ROOT))

import numpy as np
import torch

from ballontranslator.modules.ocr import OCR
from ballontranslator.utils import config as program_config
from ballontranslator.utils import io_utils
from ballontranslator.utils.textblock import TextBlock

OUT_DIR = ROOT / 'outputs'
DEFAULT_IMAGE = ROOT / 'data' / 'p1.jpg'
GT_DIR = ROOT / 'samples'
DEFAULT_ENGINES = ['mit48px', 'manga_ocr', 'PaddleOCRVLManga', 'ppv6_onnx']


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (INVOKE_CWD / path).resolve()


def load_blocks(blocks_json: Path) -> list:
    blocks = []
    for r in json.loads(blocks_json.read_text(encoding='utf-8')):
        blk = TextBlock(
            xyxy=r['xyxy'],
            lines=[np.asarray(line, dtype=np.float32) for line in r['lines']],
            angle=r.get('angle', 0),
            src_is_vertical=r.get('src_is_vertical', False),
        )
        blk._detected_font_size = r.get('detected_font_size', r.get('font_size', -1))
        blocks.append(blk)
    return blocks


def build_ocr(key: str, device: str):
    entry = OCR.get(key)
    if entry is None:
        raise SystemExit(f'未知 OCR：{key}，可用：{", ".join(OCR.module_dict.keys())}')
    ocr_cls = entry.resolve() if hasattr(entry, 'resolve') else entry
    params = program_config.pcfg.module.get_params('ocr').get(key) or {}
    ocr = ocr_cls(**params)
    if device != 'auto':
        ocr.set_param_value('device', device)
    return ocr


def normalize(s: str) -> str:
    s = unicodedata.normalize('NFKC', s or '')
    s = ''.join(ch for ch in s if not ch.isspace())
    # 行尾的破折号/省略号/标点属于排版差异，不计入字符错误率
    return s.rstrip('。、，．,.！？!?…‥・—-―—～~ー:：;；')


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    g = normalize(gt)
    return edit_distance(normalize(pred), g) / max(1, len(g))


def main():
    parser = argparse.ArgumentParser(description='多 OCR 对比 + CER 打分')
    parser.add_argument('--image', default=None, help='原图，默认 data/p1.jpg')
    parser.add_argument('--blocks', default=None, help='文字块 json，默认 outputs/<stem>_blocks.json')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--engines', nargs='+', default=DEFAULT_ENGINES)
    args = parser.parse_args()

    image_path = resolve_path(args.image) if args.image else DEFAULT_IMAGE
    stem = image_path.stem
    blocks_path = resolve_path(args.blocks) if args.blocks else OUT_DIR / f'{stem}_blocks.json'
    if not image_path.exists() or not blocks_path.exists():
        raise SystemExit('先跑第 1 步生成 *_blocks.json')

    gt = {}
    gt_path = GT_DIR / f'{stem}_gt.json'  # 标注按图片名区分，p1_gt.json / p2_gt.json
    if gt_path.exists():
        raw = json.loads(gt_path.read_text(encoding='utf-8'))
        # 一个块可以有多个可接受答案（例如注音可有可无、字形有歧义）
        gt = {int(k): (v if isinstance(v, list) else [v]) for k, v in raw.items()}
    else:
        print(f'[提示] 没有标注 {gt_path.name}，只输出结果、不打分')

    program_config.load_config()
    img = io_utils.imread(str(image_path))
    results = {}

    for key in args.engines:
        print(f'\n=== {key} ===', flush=True)
        blocks = load_blocks(blocks_path)  # 每个引擎都用全新的 TextBlock，避免互相污染
        try:
            ocr = build_ocr(key, args.device)
            t0 = time.time()
            ret = ocr.run_ocr(img, blocks)
            cost = time.time() - t0
            if isinstance(ret, list):
                blocks = ret
        except Exception as e:
            print(f'  失败：{type(e).__name__}: {e}', flush=True)
            results[key] = {'error': f'{type(e).__name__}: {e}'}
            continue

        preds = {i: blk.get_text() for i, blk in enumerate(blocks)}
        scores = {}
        for bid, refs in gt.items():
            p = preds.get(bid, '')
            best = min(refs, key=lambda ref: cer(p, ref))
            scores[bid] = {
                'pred': p,
                'gt': best,
                'alts': refs,
                'cer': round(cer(p, best), 4),
            }
        exact = sum(1 for s in scores.values() if s['cer'] == 0)
        mean_cer = (sum(s['cer'] for s in scores.values()) / len(scores)) if scores else None

        print(f'  耗时 {cost:.1f}s', flush=True)
        for bid in range(len(blocks)):
            marker = ''
            if bid in scores:
                marker = f"  CER {scores[bid]['cer'] * 100:5.1f}%"
            print(f"  #{bid}{marker}  {preds.get(bid, '')}", flush=True)
        if mean_cer is not None:
            print(f'  >>> 平均 CER {mean_cer * 100:.1f}% | 完全正确 {exact}/{len(scores)}', flush=True)

        results[key] = {
            'time': round(cost, 2),
            'mean_cer': round(mean_cer, 4) if mean_cer is not None else None,
            'exact': f'{exact}/{len(scores)}' if scores else None,
            'preds': {str(k): v for k, v in preds.items()},
            'scores': {str(k): v for k, v in scores.items()},
        }

        try:
            ocr.unload_model(empty_cache=True)
        except Exception:
            pass
        del ocr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{stem}_ocr_compare.json'
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n===== 汇总 =====')
    print(f"{'engine':18s} {'mean CER':>9s} {'exact':>7s} {'time':>7s}")
    for key, r in results.items():
        if 'error' in r:
            print(f'{key:18s} {"FAILED":>9s}')
        else:
            mc = f"{r['mean_cer'] * 100:.1f}%" if r['mean_cer'] is not None else '-'
            print(f"{key:18s} {mc:>9s} {str(r['exact']):>7s} {r['time']:>6.1f}s")
    print(f'\n明细 -> {out_path}')


if __name__ == '__main__':
    main()
