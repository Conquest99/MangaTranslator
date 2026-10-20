"""第 3 步：翻译（复用 BallonsTranslator 的翻译器框架）。

第 2 步把文字读出来了，这一步把它变成译文：

    List[TextBlock].text  ──►  List[TextBlock].translation

TextBlock 上 text 和 translation 是**两个独立字段**，原文一直保留，
所以后面随时可以对照、重翻、或者只改嵌字不重翻。

三种翻译器，接口完全一样，只是背后不同：
    Copy Source     原样返回，用来打通链路（不需要任何模型/API）
    LLMTranslator   OpenAI 兼容接口，云端或本地服务都行（需在 config 里配 profile）
    Sakura          日→中专用模型，指向本地 llama.cpp / LM Studio 的 OpenAI 兼容服务

用法：
    整合\\.venv\\Scripts\\python.exe 整合\\引入翻译.py
    整合\\.venv\\Scripts\\python.exe 整合\\引入翻译.py --translator LLMTranslator
    整合\\.venv\\Scripts\\python.exe 整合\\引入翻译.py --ocr-json 整合\\outputs\\001_ocr.json
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

import numpy as np

from ballontranslator.modules.translators import TRANSLATORS
from ballontranslator.utils import config as program_config
from ballontranslator.utils.textblock import TextBlock

# 导入即把 'DeepSeek API' 注册进 TRANSLATORS
import trans_api  # noqa: F401

DEFAULT_OCR_JSON = ROOT / 'outputs' / 'sample_page_ocr.json'
OUT_DIR = ROOT / 'outputs'


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (INVOKE_CWD / path).resolve()


def load_blocks(ocr_json: Path) -> list:
    """从第 2 步的 json 还原 TextBlock（原文 + 几何 + 颜色）。"""
    records = json.loads(ocr_json.read_text(encoding='utf-8'))
    blocks = []
    for r in records:
        blk = TextBlock(
            xyxy=r['xyxy'],
            lines=[np.asarray(line, dtype=np.float32) for line in r['lines']],
            angle=r.get('angle', 0),
            src_is_vertical=r.get('src_is_vertical', False),
            text=list(r['text_lines']),
        )
        blk._detected_font_size = r.get('font_size', -1)
        blocks.append(blk)
    return blocks


def build_translator(key: str, source: str, target: str, overrides: dict = None):
    entry = TRANSLATORS.get(key)
    if entry is None:
        available = ', '.join(TRANSLATORS.module_dict.keys())
        raise SystemExit(f'没有名为 {key!r} 的翻译器。可用：{available}')

    translator_cls = entry.resolve() if hasattr(entry, 'resolve') else entry
    # 配置里保存过的参数（API key / baseurl / 温度等），没有就用默认值
    params = dict(program_config.pcfg.module.get_params('translator').get(key) or {})
    if overrides:
        params.update(overrides)
    translator = translator_cls(source, target, raise_unsupported_lang=False, **params)
    translator.load_model()
    return translator


def main():
    parser = argparse.ArgumentParser(description='第 3 步：翻译')
    parser.add_argument('--ocr-json', default=None, help='第 2 步的 *_ocr.json，默认 sample_page_ocr.json')
    parser.add_argument('--translator', default='Copy Source', help="注册表里的 key，如 'Copy Source' / 'DeepSeek API' / 'Sakura'")
    parser.add_argument('--source', default=None, help='源语言，默认读 config')
    parser.add_argument('--target', default=None, help='目标语言，默认读 config')
    parser.add_argument('--sakura-version', default=None, choices=['0.9', '1.0', 'galtransl-v1'],
                        help='Sakura 提示词版本：v3.x Galtransl 模型用 galtransl-v1')
    parser.add_argument('--api-model', default=None, help="API 模型名，如 deepseek-chat / deepseek-reasoner")
    args = parser.parse_args()

    ocr_json = resolve_path(args.ocr_json) if args.ocr_json else DEFAULT_OCR_JSON
    if not ocr_json.exists():
        raise SystemExit(f'找不到 OCR 结果：{ocr_json}\n先跑第 2 步：整合\\引入ORC.py')

    program_config.load_config()
    source = args.source or program_config.pcfg.module.translate_source
    target = args.target or program_config.pcfg.module.translate_target

    blocks = load_blocks(ocr_json)
    print(f'从 {ocr_json.name} 载入 {len(blocks)} 个文字块')

    overrides = None
    if args.sakura_version:
        overrides = {'version': args.sakura_version}
    elif args.api_model:
        overrides = {'model': args.api_model}
    translator = build_translator(args.translator, source, target, overrides)
    print(f'翻译器 {translator.name} | {translator.lang_source} -> {translator.lang_target}')

    t0 = time.time()
    # 注意：这里传的是 TextBlock 列表，不是字符串列表。
    # 空块会被跳过；project/page_key 用于 LLM 的上下文/术语表，这里没有项目所以留空。
    translator.translate_textblk_lst(blocks, project=None, page_key=None)
    cost = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = ocr_json.stem.removesuffix('_ocr')
    json_path = OUT_DIR / f'{stem}_trans.json'
    txt_path = OUT_DIR / f'{stem}_trans.txt'

    records = []
    for i, blk in enumerate(blocks):
        records.append({
            'id': i,
            'xyxy': [int(v) for v in blk.xyxy],
            'src_is_vertical': bool(blk.src_is_vertical),
            'font_size': round(float(blk._detected_font_size), 2),
            'source': blk.get_text(),
            'translation': blk.translation,
            'lines': [np.asarray(line).round(1).tolist() for line in blk.lines],
        })
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    txt_lines = []
    print(f'\n翻译耗时 {cost:.2f}s：')
    for r in records:
        print(f"  #{r['id']} {r['source']}")
        print(f"      -> {r['translation']}")
        txt_lines.append(f"#{r['id']}\t{r['source']}\t{r['translation']}")
    txt_path.write_text('\n'.join(txt_lines) + '\n', encoding='utf-8')

    print(f'\n已保存：\n  对照 -> {txt_path}\n  数据 -> {json_path}')


if __name__ == '__main__':
    main()
