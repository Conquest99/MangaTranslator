"""批量流水线后端：图片 / 文件夹 / .cbz → 检测 → OCR → 翻译 → 抹字 → 嵌字。

不依赖 Qt，命令行和 GUI 共用。每一步都复用 整合/ 下已经验证过的阶段脚本，
所以行为和单步跑完全一致。

    from pipeline import Pipeline, Options, collect_pages
    pipe = Pipeline(Options(...))
    pipe.run(collect_pages([Path('整合/data')]))
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent
BT_ROOT = (ROOT.parent / 'BallonsTranslator').resolve()
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.chdir(BT_ROOT)  # BallonsTranslator 用相对路径找 config/ 和 data/models/
sys.path.insert(0, str(BT_ROOT))

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image, ImageDraw

from ballontranslator.utils import config as program_config
from ballontranslator.utils import io_utils

IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
DEFAULT_FONT = r'C:\Windows\Fonts\msyh.ttc'


def _load_stage(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 复用已经验证过的阶段脚本；模块级只有路径设置，import 没有副作用
stage_detect = _load_stage('stage_detect', '引入引擎.py')
stage_ocr = _load_stage('stage_ocr', '引入ORC.py')
stage_mt = _load_stage('stage_mt', '引入翻译.py')
stage_inpaint = _load_stage('stage_inpaint', '擦除.py')
stage_render = _load_stage('stage_render', '渲染嵌字.py')
# 导入即把 'DeepSeek API' 注册进 TRANSLATORS
stage_api = _load_stage('trans_api', 'trans_api.py')


@dataclass
class Options:
    out_dir: Path = ROOT / 'outputs' / 'gui'
    ocr_engine: str = 'PaddleOCRVLManga'
    translator: str = 'Sakura'
    sakura_version: str = 'galtransl-v1'
    api_model: str = 'deepseek-chat'
    api_base_url: str = 'https://api.deepseek.com/v1'
    inpainter: str = 'lama_large_512px'
    inpaint_size: int = 1024
    device: str = 'auto'
    detect_size: int = 1280
    source_lang: str = '日本語'
    target_lang: str = '简体中文'
    font: str = DEFAULT_FONT
    do_detect: bool = True
    do_ocr: bool = True
    do_translate: bool = True
    do_inpaint: bool = True
    do_render: bool = True


@dataclass
class PageSource:
    display: str
    image_path: Optional[Path] = None
    cbz_path: Optional[Path] = None
    entry: Optional[str] = None


def _cbz_entries(cbz_path: Path) -> List[str]:
    with zipfile.ZipFile(cbz_path) as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(IMG_EXT) and not n.endswith('/')
        ]
    return natsorted(names)


def collect_pages(paths: List[Path]) -> List[PageSource]:
    pages: List[PageSource] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for f in natsorted(path.iterdir(), key=lambda p: p.name):
                if f.suffix.lower() in IMG_EXT:
                    pages.append(PageSource(display=f'{path.name}/{f.name}', image_path=f))
                elif f.suffix.lower() == '.cbz':
                    stem = f.stem
                    pages.extend(
                        PageSource(display=f'{stem}/{name}', cbz_path=f, entry=name)
                        for name in _cbz_entries(f)
                    )
        elif path.suffix.lower() == '.cbz':
            pages.extend(
                PageSource(display=f'{path.stem}/{name}', cbz_path=path, entry=name)
                for name in _cbz_entries(path)
            )
        elif path.suffix.lower() in IMG_EXT:
            pages.append(PageSource(display=path.name, image_path=path))
    return pages


def read_page_image(src: PageSource) -> np.ndarray:
    if src.image_path is not None:
        return io_utils.imread(str(src.image_path))
    with zipfile.ZipFile(src.cbz_path) as z:
        data = z.read(src.entry)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'无法解码 {src.display}')
    return img


def page_output_path(out_dir: Path, src: PageSource) -> Path:
    book = (src.cbz_path.stem if src.cbz_path else 'images')
    stem = Path(src.entry).stem if src.entry else src.image_path.stem
    return out_dir / book / f'{stem}.png'


class Pipeline:
    """顺序处理每一页；回调都在调用线程里触发（GUI 里由 QThread 发出信号）。"""

    def __init__(
        self,
        options: Options,
        on_log: Optional[Callable[[str], None]] = None,
        on_page_started: Optional[Callable[[int, PageSource], None]] = None,
        on_stage: Optional[Callable[[int, str], None]] = None,
        on_page_done: Optional[Callable[[int, PageSource, Path, Optional[Path]], None]] = None,
        on_error: Optional[Callable[[int, PageSource, str], None]] = None,
    ) -> None:
        self.opt = options
        self.on_log = on_log or (lambda m: None)
        self.on_page_started = on_page_started or (lambda i, p: None)
        self.on_stage = on_stage or (lambda i, s: None)
        self.on_page_done = on_page_done or (lambda i, p, f, m: None)
        self.on_error = on_error or (lambda i, p, e: None)
        self._stop = threading.Event()
        self.detector = None
        self.ocr = None
        self.translator = None
        self.inpainter = None

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ---------- 模型 ----------

    def _ensure_models(self) -> None:
        if self.opt.do_detect and self.detector is None:
            self.on_log('加载文本检测模型...')
            self.detector = stage_detect.build_detector(self.opt.device, self.opt.detect_size)
        if self.opt.do_ocr and self.ocr is None:
            self.on_log(f'加载 OCR：{self.opt.ocr_engine} ...')
            self.ocr = stage_ocr.build_ocr(self.opt.ocr_engine, self.opt.device)
        if self.opt.do_translate and self.translator is None:
            self.on_log(f'加载翻译器：{self.opt.translator} ...')
            overrides = None
            if self.opt.translator == 'Sakura':
                overrides = {'version': self.opt.sakura_version}
            elif self.opt.translator == 'DeepSeek API':
                overrides = {
                    'model': self.opt.api_model,
                    'api baseurl': self.opt.api_base_url,
                }
            self.translator = stage_mt.build_translator(
                self.opt.translator, self.opt.source_lang, self.opt.target_lang, overrides
            )
        if self.opt.do_inpaint and self.inpainter is None:
            self.on_log(f'加载修复器：{self.opt.inpainter} ...')
            self.inpainter = stage_inpaint.build_inpainter(
                self.opt.inpainter, self.opt.device, self.opt.inpaint_size
            )

    # ---------- 单页 ----------

    def process_page(self, index: int, src: PageSource):
        opt = self.opt
        img = read_page_image(src)
        mask = None
        blocks = []

        if opt.do_detect:
            self.on_stage(index, 'detect')
            mask, blocks = self.detector.detect(img)
            self.on_log(f'[{index + 1}] {src.display}: 检测到 {len(blocks)} 块')

        if opt.do_ocr and blocks:
            self.on_stage(index, 'ocr')
            ret = self.ocr.run_ocr(img, blocks)
            if isinstance(ret, list):
                blocks = ret
            self.on_log(f'[{index + 1}] OCR 完成')

        if opt.do_translate and blocks:
            self.on_stage(index, 'translate')
            self.translator.translate_textblk_lst(blocks, project=None, page_key=None)
            self.on_log(f'[{index + 1}] 翻译完成')

        base = img
        inpainted_path: Optional[Path] = None
        out_path = page_output_path(opt.out_dir, src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if opt.do_inpaint and mask is not None and blocks:
            self.on_stage(index, 'inpaint')
            base = self.inpainter.inpaint(img, mask.copy(), blocks)
            self.on_log(f'[{index + 1}] 抹字完成')
            inpainted_path = out_path.with_name(f'{out_path.stem}_inpainted.png')
            Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)).save(inpainted_path)

        if opt.do_render and blocks:
            self.on_stage(index, 'render')
            canvas = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
            render_blocks(canvas, blocks, opt.font)
            canvas.save(out_path)
        else:
            Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)).save(out_path)

        meta = {
            'page': src.display,
            'blocks': [
                {
                    'id': i,
                    'xyxy': [int(v) for v in blk.xyxy],
                    'text': blk.get_text(),
                    'translation': blk.translation,
                    'vertical': bool(blk.vertical),
                    'font_size': round(float(blk._detected_font_size), 2),
                }
                for i, blk in enumerate(blocks)
            ],
        }
        out_path.with_suffix('.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        return out_path, inpainted_path

    # ---------- 批量 ----------

    def run(self, pages: List[PageSource]) -> None:
        program_config.load_config()
        self.on_log(f'共 {len(pages)} 页，开始处理')
        self._ensure_models()
        for index, src in enumerate(pages):
            if self.stopped:
                self.on_log('已停止')
                break
            self.on_page_started(index, src)
            try:
                out, inpainted = self.process_page(index, src)
                if out is not None:
                    self.on_page_done(index, src, out, inpainted)
            except Exception as e:  # 单页失败不影响后续
                self.on_error(index, src, f'{type(e).__name__}: {e}')
        self.on_log('全部完成' if not self.stopped else '已停止')


def render_blocks(canvas: Image.Image, blocks, font_path: str) -> None:
    """复用 渲染嵌字.py 的排版函数。"""
    draw = ImageDraw.Draw(canvas)
    for blk in blocks:
        text = (blk.translation or blk.get_text() or '').strip()
        if not text:
            continue
        x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 2 or box_h <= 2:
            continue
        base_size = float(blk._detected_font_size)
        if base_size <= 0:
            base_size = 24.0
        try:
            fg, _bg = blk.get_font_colors()
            fill = stage_render.text_color(tuple(int(v) for v in fg))
        except Exception:
            fill = (0, 0, 0)

        if blk.vertical:
            font, cols = stage_render.fit_vertical(text, box_w, box_h, base_size, font_path)
            stroke = max(1, round(font.size / 16))
            stage_render.draw_vertical(draw, cols, [x1, y1, x2, y2], font, fill, stroke)
        else:
            font, lines = stage_render.fit_horizontal(text, draw, box_w, box_h, base_size, font_path)
            stroke = max(1, round(font.size / 16))
            stage_render.draw_horizontal(draw, lines, [x1, y1, x2, y2], font, fill, stroke)
