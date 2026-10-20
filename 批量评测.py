"""批量 OCR 评测：.cbz → 检测缓存 → 多引擎 OCR → 共识 / 标注 CER。

.cbz 就是 zip，直接读图不解压。

两套指标：
    共识评测（无需标注，覆盖全量）
        对每个文字块，取各引擎归一化结果的多数票作为伪参考，
        算每个引擎离共识有多远 + 空输出率 + 截断率。
    标注评测（绝对准确）
        如果 samples/batch_gt/<cbz名>__<页名>.json 存在，就算真正的 CER。
        用 --dump-label N 可以导出前 N 页的裁剪图和空模板，方便手工标注。

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\批量评测.py --cbz 整合\\data
    整合\\.venv\\Scripts\\python.exe 整合\\批量评测.py --cbz 整合\\data --dump-label 10
    整合\\.venv\\Scripts\\python.exe 整合\\批量评测.py --cbz 整合\\data --engines mit48px manga_ocr
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BT_ROOT = (ROOT.parent / 'BallonsTranslator').resolve()
INVOKE_CWD = Path.cwd()
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.chdir(BT_ROOT)
sys.path.insert(0, str(BT_ROOT))

import cv2
import numpy as np
import torch
from natsort import natsorted

from ballontranslator.modules.ocr import OCR
from ballontranslator.utils import config as program_config
from ballontranslator.utils import io_utils
from ballontranslator.utils.textblock import TextBlock

OUT_DIR = ROOT / 'outputs' / 'batch'
GT_DIR = ROOT / 'samples' / 'batch_gt'
IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
DEFAULT_ENGINES = ['mit48px', 'manga_ocr', 'PaddleOCRVLManga', 'ppv6_onnx']


# ---------- 基础工具 ----------

def natural_key(name: str):
    return natsorted([name])


def list_pages(cbz_path: Path):
    with zipfile.ZipFile(cbz_path) as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(IMG_EXT) and not n.endswith('/')
        ]
    return natsorted(names)


def read_cbz_image(cbz_path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(cbz_path) as z:
        data = z.read(name)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'无法解码 {cbz_path.name}:{name}')
    return img


def normalize(s: str) -> str:
    s = unicodedata.normalize('NFKC', s or '')
    s = ''.join(ch for ch in s if not ch.isspace())
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


def cer(pred: str, ref: str) -> float:
    r = normalize(ref)
    return edit_distance(normalize(pred), r) / max(1, len(r))


def page_id(cbz_stem: str, page_name: str) -> str:
    return f'{cbz_stem}__{Path(page_name).stem}'


# ---------- 检测与 OCR ----------

def load_stage1():
    import importlib.util
    spec = importlib.util.spec_from_file_location('stage1_detector', ROOT / '引入引擎.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocks_to_records(blocks) -> list:
    return [
        {
            'id': i,
            'xyxy': [int(v) for v in blk.xyxy],
            'angle': int(blk.angle),
            'src_is_vertical': bool(blk.src_is_vertical),
            'detected_font_size': round(float(blk._detected_font_size), 2),
            'lines': [np.asarray(line).round(1).tolist() for line in blk.lines],
        }
        for i, blk in enumerate(blocks)
    ]


def records_to_blocks(records):
    blocks = []
    for r in records:
        blk = TextBlock(
            xyxy=r['xyxy'],
            lines=[np.asarray(line, dtype=np.float32) for line in r['lines']],
            angle=r.get('angle', 0),
            src_is_vertical=r.get('src_is_vertical', False),
        )
        blk._detected_font_size = r.get('detected_font_size', -1)
        blocks.append(blk)
    return blocks


def build_ocr(key: str, device: str):
    entry = OCR.get(key)
    if entry is None:
        raise SystemExit(f'未知 OCR：{key}')
    ocr_cls = entry.resolve() if hasattr(entry, 'resolve') else entry
    params = program_config.pcfg.module.get_params('ocr').get(key) or {}
    ocr = ocr_cls(**params)
    if device != 'auto':
        ocr.set_param_value('device', device)
    return ocr


def main():
    parser = argparse.ArgumentParser(description='批量 OCR 评测（cbz）')
    parser.add_argument('--cbz', required=True, help='单个 .cbz 或包含 .cbz 的目录')
    parser.add_argument('--engines', nargs='+', default=DEFAULT_ENGINES)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--pages', default='all', help='all 或 1-10 / 1,3,5')
    parser.add_argument('--detect-size', type=int, default=1280)
    parser.add_argument('--force-detect', action='store_true', help='忽略检测缓存')
    parser.add_argument('--dump-label', type=int, default=0, help='导出前 N 页的裁剪图和标注模板')
    args = parser.parse_args()

    src = Path(args.cbz)
    if not src.is_absolute():
        src = (INVOKE_CWD / src).resolve()
    cbz_files = [src] if src.is_file() else natsorted(str(p) for p in src.glob('*.cbz'))
    cbz_files = [Path(p) for p in cbz_files]
    if not cbz_files:
        raise SystemExit(f'没有找到 .cbz：{src}')

    page_range = None
    if args.pages != 'all':
        page_range = set()
        for part in args.pages.split(','):
            if '-' in part:
                a, b = part.split('-')
                page_range.update(range(int(a), int(b) + 1))
            else:
                page_range.add(int(part))

    program_config.load_config()
    stage1 = load_stage1()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有要处理的页
    jobs = []  # (cbz_path, page_name)
    for cbz in cbz_files:
        pages = list_pages(cbz)
        for ii, name in enumerate(pages, 1):
            if page_range is None or ii in page_range:
                jobs.append((cbz, name, ii))
    print(f'共 {len(cbz_files)} 个 cbz，{len(jobs)} 页待处理', flush=True)

    # ---------- 阶段 1：检测（缓存） ----------
    detector = stage1.build_detector(args.device, args.detect_size)
    for cbz, page, ii in jobs:
        stem = cbz.stem
        cache = OUT_DIR / stem / f'{Path(page).stem}_blocks.json'
        if cache.exists() and not args.force_detect:
            continue
        img = read_cbz_image(cbz, page)
        _, blocks = detector.detect(img)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(blocks_to_records(blocks), ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'[detect] {stem}:{page}  {len(blocks)} 块', flush=True)
    try:
        detector.unload_model(empty_cache=True)
    except Exception:
        pass
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- 阶段 2：每个引擎跑全部页 ----------
    timings = {}
    for engine in args.engines:
        t_engine = time.time()
        print(f'\n=== {engine} ===', flush=True)
        ocr = build_ocr(engine, args.device)
        for cbz, page, ii in jobs:
            stem = cbz.stem
            blocks_path = OUT_DIR / stem / f'{Path(page).stem}_blocks.json'
            out_path = OUT_DIR / stem / f'{Path(page).stem}_ocr_{engine}.json'
            if out_path.exists():
                continue
            records = json.loads(blocks_path.read_text(encoding='utf-8'))
            blocks = records_to_blocks(records)
            img = read_cbz_image(cbz, page)
            try:
                ret = ocr.run_ocr(img, blocks)
                if isinstance(ret, list):
                    blocks = ret
                preds = {i: blk.get_text() for i, blk in enumerate(blocks)}
                payload = {'engine': engine, 'page': page, 'preds': preds}
            except Exception as e:
                payload = {'engine': engine, 'page': page, 'error': f'{type(e).__name__}: {e}'}
            out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            print(f'  {page}', flush=True)
        timings[engine] = time.time() - t_engine
        try:
            ocr.unload_model(empty_cache=True)
        except Exception:
            pass
        del ocr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 阶段 3：汇总 ----------
    stats = {
        e: {'blocks': 0, 'cer_sum': 0.0, 'exact': 0, 'empty': 0, 'trunc': 0,
            'gt_blocks': 0, 'gt_cer_sum': 0.0, 'gt_exact': 0}
        for e in args.engines
    }
    details = []

    for cbz, page, ii in jobs:
        stem = cbz.stem
        blocks = json.loads((OUT_DIR / stem / f'{Path(page).stem}_blocks.json').read_text(encoding='utf-8'))
        preds = {}
        for engine in args.engines:
            p = OUT_DIR / stem / f'{Path(page).stem}_ocr_{engine}.json'
            if not p.exists():
                preds[engine] = {}
                continue
            payload = json.loads(p.read_text(encoding='utf-8'))
            preds[engine] = payload.get('preds', {})

        gt = {}
        gt_path = GT_DIR / f'{page_id(stem, page)}.json'
        if gt_path.exists():
            raw = json.loads(gt_path.read_text(encoding='utf-8'))
            gt = {int(k): (v if isinstance(v, list) else [v]) for k, v in raw.items() if v}

        for bid in range(len(blocks)):
            texts = {e: preds[e].get(str(bid), preds[e].get(bid, '')) for e in args.engines}
            norms = {e: normalize(t) for e, t in texts.items()}
            cands = [n for n in norms.values() if n]
            if not cands:
                continue
            # 多数票做伪参考；平票取最长
            cnt = Counter(cands)
            consensus = max(cnt.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
            max_len = max(len(n) for n in cands)

            for e in args.engines:
                n = norms[e]
                st = stats[e]
                st['blocks'] += 1
                if not n:
                    st['empty'] += 1
                if max_len and len(n) < 0.6 * max_len:
                    st['trunc'] += 1
                st['cer_sum'] += edit_distance(n, consensus) / max(1, len(consensus))
                if n == consensus:
                    st['exact'] += 1

                if bid in gt:
                    best = min(gt[bid], key=lambda ref: cer(texts[e], ref))
                    c = cer(texts[e], best)
                    st['gt_blocks'] += 1
                    st['gt_cer_sum'] += c
                    if c == 0:
                        st['gt_exact'] += 1

            details.append({
                'page': f'{stem}:{page}', 'block': bid,
                'texts': texts, 'consensus': consensus,
                'gt': gt.get(bid),
            })

    # 导出裁剪图 + 空标注模板（可选）
    if args.dump_label > 0:
        GT_DIR.mkdir(parents=True, exist_ok=True)
        for cbz, page, ii in jobs[:args.dump_label]:
            stem = cbz.stem
            key = page_id(stem, page)
            records = json.loads((OUT_DIR / stem / f'{Path(page).stem}_blocks.json').read_text(encoding='utf-8'))
            img = read_cbz_image(cbz, page)
            crop_dir = OUT_DIR / 'label_crops' / key
            crop_dir.mkdir(parents=True, exist_ok=True)
            template = {}
            for r in records:
                bid = r['id']
                x1, y1, x2, y2 = r['xyxy']
                p = 12
                crop = img[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
                io_utils.imwrite(str(crop_dir / f'blk{bid:02d}.png'), crop)
                template[str(bid)] = ''
            tpl = GT_DIR / f'{key}.json'
            if not tpl.exists():
                tpl.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n标注模板已写到 {GT_DIR}', flush=True)

    # ---------- 报告 ----------
    lines = ['# OCR 批量评测', '']
    lines.append(f'页面数：{len(jobs)} | 块数（任一引擎非空）：{len(details)}')
    lines.append('')
    lines.append('## 共识评测（全部块）')
    lines.append('')
    lines.append('| 引擎 | CER vs 共识 | 与共识完全一致 | 空输出率 | 截断率 | 耗时 |')
    lines.append('|---|---|---|---|---|---|')
    for e in args.engines:
        st = stats[e]
        n = max(1, st['blocks'])
        lines.append(
            f"| {e} | {st['cer_sum'] / n * 100:.1f}% | {st['exact'] / n * 100:.1f}% | "
            f"{st['empty'] / n * 100:.1f}% | {st['trunc'] / n * 100:.1f}% | "
            f"{timings.get(e, 0):.0f}s |"
        )
    lines.append('')
    lines.append('## 标注评测（有 ground truth 的块）')
    lines.append('')
    lines.append('| 引擎 | 平均 CER | 完全正确 | 标注块数 |')
    lines.append('|---|---|---|---|')
    for e in args.engines:
        st = stats[e]
        if st['gt_blocks'] == 0:
            lines.append(f'| {e} | - | - | 0 |')
        else:
            lines.append(
                f"| {e} | {st['gt_cer_sum'] / st['gt_blocks'] * 100:.1f}% | "
                f"{st['gt_exact']}/{st['gt_blocks']} | {st['gt_blocks']} |"
            )
    report = '\n'.join(lines)
    (OUT_DIR / 'report.md').write_text(report, encoding='utf-8')
    (OUT_DIR / 'details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n' + report)
    print(f'\n报告 -> {OUT_DIR / "report.md"}\n明细 -> {OUT_DIR / "details.json"}')


if __name__ == '__main__':
    main()
