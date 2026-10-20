"""批量漫画翻译 - PyQt6 图形界面。

    python 整合/app.py

流程：添加图片 / 文件夹 / .cbz → 选好引擎 → 开始 → 输出到 outputs/gui。

界面布局参考 BallonsTranslator：左侧页面列表、中间预览、右侧设置、底部日志与进度。
注意：Sakura 翻译需要先在另一个终端启动 llama-server（见 README/对话记录）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import Options, PageSource, Pipeline, collect_pages, read_page_image  # noqa: E402

STAGE_NAMES = {
    'detect': '检测',
    'ocr': 'OCR',
    'translate': '翻译',
    'inpaint': '抹字',
    'render': '嵌字',
}
IMAGE_FILTER = '漫画 (*.cbz *.jpg *.jpeg *.png *.webp *.bmp);;所有文件 (*)'


def to_pixmap(img: np.ndarray, max_width: int = 900) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if w > max_width:
        pix = pix.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
    return pix


class Worker(QThread):
    log = pyqtSignal(str)
    page_started = pyqtSignal(int, str)
    stage = pyqtSignal(int, str)
    page_done = pyqtSignal(int, str, str)
    page_error = pyqtSignal(int, str, str)
    done = pyqtSignal()

    def __init__(self, options: Options, pages: list[PageSource]) -> None:
        super().__init__()
        self.options = options
        self.pages = pages
        self.pipeline: Pipeline | None = None

    def run(self) -> None:
        self.pipeline = Pipeline(
            self.options,
            on_log=self.log.emit,
            on_page_started=lambda i, p: self.page_started.emit(i, p.display),
            on_stage=lambda i, s: self.stage.emit(i, s),
            on_page_done=lambda i, p, f, m: self.page_done.emit(i, str(f), str(m) if m else ''),
            on_error=lambda i, p, e: self.page_error.emit(i, p.display, e),
        )
        self.pipeline.run(self.pages)
        self.done.emit()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('MangaTransfer - 批量漫画翻译')
        self.resize(1500, 950)
        self.pages: list[PageSource] = []
        self.outputs: dict[int, Path] = {}
        self.inpainted: dict[int, Path] = {}
        self.worker: Worker | None = None
        self.current_index = -1
        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        toolbar = QToolBar('主工具栏')
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)

        act_files = QAction('添加图片 / CBZ', self)
        act_files.triggered.connect(self.add_files)
        act_folder = QAction('添加文件夹', self)
        act_folder.triggered.connect(self.add_folder)
        act_clear = QAction('清空列表', self)
        act_clear.triggered.connect(self.clear_pages)
        self.act_start = QAction('开始', self)
        self.act_start.triggered.connect(self.start)
        self.act_stop = QAction('停止', self)
        self.act_stop.triggered.connect(self.stop)
        self.act_stop.setEnabled(False)
        act_open = QAction('打开输出目录', self)
        act_open.triggered.connect(self.open_output)

        for act in (act_files, act_folder, act_clear):
            toolbar.addAction(act)
        toolbar.addSeparator()
        toolbar.addAction(self.act_start)
        toolbar.addAction(self.act_stop)
        toolbar.addSeparator()
        toolbar.addAction(act_open)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.page_list = QListWidget()
        self.page_list.setMinimumWidth(280)
        self.page_list.currentRowChanged.connect(self.on_page_selected)
        splitter.addWidget(self.page_list)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(['成品（嵌字后）', '原图', '抹字后'])
        self.preview_mode.currentIndexChanged.connect(lambda _: self.refresh_preview())
        center_layout.addWidget(self.preview_mode)

        self.preview = QLabel('添加文件后在这里预览')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet('background:#2b2b2b;color:#aaa;')
        scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(True)
        center_layout.addWidget(scroll, 1)
        splitter.addWidget(center)

        splitter.addWidget(self.build_settings())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 900, 320])

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(6, 6, 6, 6)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(140)
        bottom_layout.addWidget(self.progress)
        bottom_layout.addWidget(self.log)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(bottom)
        self.setCentralWidget(root)

    def build_settings(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)

        out_box = QGroupBox('输出')
        out_form = QFormLayout(out_box)
        self.out_dir_edit = QLineEdit(str(ROOT / 'outputs' / 'gui'))
        btn_browse = QPushButton('选择…')
        btn_browse.clicked.connect(self.choose_output)
        row = QHBoxLayout()
        row.addWidget(self.out_dir_edit, 1)
        row.addWidget(btn_browse)
        out_form.addRow('目录', row)
        layout.addWidget(out_box)

        engine_box = QGroupBox('引擎')
        engine_form = QFormLayout(engine_box)
        self.ocr_combo = QComboBox()
        self.ocr_combo.addItems(['PaddleOCRVLManga', 'mit48px', 'manga_ocr', 'ppv6_onnx'])
        self.mt_combo = QComboBox()
        self.mt_combo.addItems(['Sakura', 'DeepSeek API', 'Copy Source'])
        self.mt_combo.currentTextChanged.connect(self._on_translator_changed)
        self.sakura_version = QComboBox()
        self.sakura_version.addItems(['galtransl-v1', '1.0', '0.9'])
        self.api_model = QComboBox()
        self.api_model.setEditable(True)
        self.api_model.addItems(['deepseek-chat', 'deepseek-reasoner'])
        self.api_base = QLineEdit('https://api.deepseek.com/v1')
        self.inpaint_combo = QComboBox()
        self.inpaint_combo.addItems(['lama_large_512px', 'aot', 'patchmatch', 'opencv-tela'])
        self.inpaint_size = QComboBox()
        self.inpaint_size.addItems(['512', '768', '1024', '1536', '2048'])
        self.inpaint_size.setCurrentText('1024')
        self.device_combo = QComboBox()
        self.device_combo.addItems(['auto', 'cuda', 'cpu'])
        self.detect_size = QComboBox()
        self.detect_size.addItems(['896', '1024', '1152', '1280'])
        self.detect_size.setCurrentText('1280')
        engine_form.addRow('OCR', self.ocr_combo)
        engine_form.addRow('翻译器', self.mt_combo)
        engine_form.addRow('Sakura 版本', self.sakura_version)
        engine_form.addRow('API 模型', self.api_model)
        engine_form.addRow('API 地址', self.api_base)
        engine_form.addRow('修复器', self.inpaint_combo)
        engine_form.addRow('修复尺寸', self.inpaint_size)
        engine_form.addRow('检测尺寸', self.detect_size)
        engine_form.addRow('设备', self.device_combo)
        layout.addWidget(engine_box)

        stage_box = QGroupBox('阶段')
        stage_layout = QVBoxLayout(stage_box)
        self.chk_detect = QCheckBox('文本检测')
        self.chk_ocr = QCheckBox('OCR')
        self.chk_mt = QCheckBox('翻译')
        self.chk_inpaint = QCheckBox('抹字')
        self.chk_render = QCheckBox('嵌字')
        for chk in (self.chk_detect, self.chk_ocr, self.chk_mt, self.chk_inpaint, self.chk_render):
            chk.setChecked(True)
            stage_layout.addWidget(chk)
        layout.addWidget(stage_box)

        hint = QLabel('Sakura 需要本地服务：\nllama-server ... --port 8080\n\n'
                      'DeepSeek key 读取顺序：\n'
                      '1. 环境变量 DEEPSEEK_API_KEY\n'
                      '2. 整合/.secrets/deepseek.key')
        hint.setStyleSheet('color:#888;')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        self._on_translator_changed(self.mt_combo.currentText())
        return panel

    def _on_translator_changed(self, name: str) -> None:
        """按翻译器切换参数可见性，避免无关选项干扰。"""
        is_sakura = name == 'Sakura'
        is_api = name == 'DeepSeek API'
        for widget in (self.sakura_version,):
            widget.setEnabled(is_sakura)
        self.api_model.setEnabled(is_api)
        self.api_base.setEnabled(is_api)

    # ---------- 页面管理 ----------

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, '添加图片或 CBZ', str(ROOT / 'data'), IMAGE_FILTER)
        if files:
            self._add_sources([Path(f) for f in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, '选择文件夹', str(ROOT / 'data'))
        if folder:
            self._add_sources([Path(folder)])

    def _add_sources(self, paths: list[Path]) -> None:
        new_pages = collect_pages(paths)
        start = len(self.pages)
        self.pages.extend(new_pages)
        for i, src in enumerate(new_pages, start):
            item = QListWidgetItem(f'{src.display}')
            item.setData(Qt.ItemDataRole.UserRole, i)
            try:
                pix = to_pixmap(read_page_image(src), max_width=64)
                item.setIcon(QPixmap(pix))
            except Exception:
                pass
            self.page_list.addItem(item)
        self.progress.setMaximum(max(1, len(self.pages)))
        self.log.appendPlainText(f'添加 {len(new_pages)} 页')
        if new_pages and self.page_list.currentRow() < 0:
            self.page_list.setCurrentRow(0)

    def clear_pages(self) -> None:
        if self.worker is not None:
            QMessageBox.warning(self, '提示', '正在处理，先停止再清空')
            return
        self.pages.clear()
        self.outputs.clear()
        self.page_list.clear()
        self.preview.setText('添加文件后在这里预览')
        self.progress.setValue(0)

    # ---------- 运行 ----------

    def build_options(self) -> Options:
        return Options(
            out_dir=Path(self.out_dir_edit.text()).expanduser(),
            ocr_engine=self.ocr_combo.currentText(),
            translator=self.mt_combo.currentText(),
            sakura_version=self.sakura_version.currentText(),
            api_model=self.api_model.currentText().strip(),
            api_base_url=self.api_base.text().strip(),
            inpainter=self.inpaint_combo.currentText(),
            inpaint_size=int(self.inpaint_size.currentText()),
            device=self.device_combo.currentText(),
            detect_size=int(self.detect_size.currentText()),
            do_detect=self.chk_detect.isChecked(),
            do_ocr=self.chk_ocr.isChecked(),
            do_translate=self.chk_mt.isChecked(),
            do_inpaint=self.chk_inpaint.isChecked(),
            do_render=self.chk_render.isChecked(),
        )

    def start(self) -> None:
        if self.worker is not None:
            return
        if not self.pages:
            QMessageBox.information(self, '提示', '先添加图片、文件夹或 CBZ')
            return
        options = self.build_options()
        if options.translator == 'Sakura':
            try:
                import requests
                requests.get('http://127.0.0.1:8080/health', timeout=2).raise_for_status()
            except Exception:
                QMessageBox.warning(
                    self, 'Sakura 未启动',
                    '检测不到 http://127.0.0.1:8080。\n\n'
                    '先在另一个终端启动：\n'
                    r'D:\Download\llama\llama-server.exe '
                    r'-m "D:\python-workplace\MangaTransfer\sakura-model\Sakura-Galtransl-7B-v3.7.gguf" '
                    '-c 4096 -t 8 --port 8080\n\n'
                    '或者把翻译器切到 DeepSeek API / Copy Source。'
                )
                return
        elif options.translator == 'DeepSeek API':
            try:
                from trans_api import DeepSeekTranslator
                DeepSeekTranslator('日本語', '简体中文').resolve_api_key()
            except Exception as e:
                QMessageBox.warning(
                    self, 'API key 未配置',
                    f'{e}\n\n'
                    '任选一种方式：\n'
                    '1. 设置环境变量 DEEPSEEK_API_KEY\n'
                    r'2. 把 key 写到 整合\.secrets\deepseek.key'
                )
                return

        self.outputs.clear()
        self.inpainted.clear()
        for i in range(self.page_list.count()):
            item = self.page_list.item(i)
            item.setText(self.pages[item.data(Qt.ItemDataRole.UserRole)].display)
        self.progress.setValue(0)
        self.progress.setMaximum(len(self.pages))
        self.log.appendPlainText('开始处理...')

        self.worker = Worker(options, list(self.pages))
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.page_started.connect(self.on_page_started)
        self.worker.stage.connect(self.on_stage)
        self.worker.page_done.connect(self.on_page_done)
        self.worker.page_error.connect(self.on_page_error)
        self.worker.done.connect(self.on_done)
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.log.appendPlainText('正在停止...')

    def on_page_started(self, index: int, display: str) -> None:
        self._set_item_status(index, '处理中')

    def on_stage(self, index: int, stage: str) -> None:
        self._set_item_status(index, STAGE_NAMES.get(stage, stage))

    def on_page_done(self, index: int, path: str, inpainted: str) -> None:
        self.outputs[index] = Path(path)
        if inpainted:
            self.inpainted[index] = Path(inpainted)
        self._set_item_status(index, '完成')
        self.progress.setValue(self.progress.value() + 1)
        self.log.appendPlainText(f'[{index + 1}] 输出 {path}')
        if index == self.current_index:
            self.refresh_preview()

    def on_page_error(self, index: int, display: str, error: str) -> None:
        self._set_item_status(index, '失败')
        self.progress.setValue(self.progress.value() + 1)
        self.log.appendPlainText(f'[{index + 1}] 失败 {display}: {error}')

    def on_done(self) -> None:
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.log.appendPlainText('任务结束')
        self.worker = None

    def _set_item_status(self, index: int, status: str) -> None:
        for i in range(self.page_list.count()):
            item = self.page_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == index:
                item.setText(f'{self.pages[index].display}  [{status}]')
                break

    # ---------- 预览 ----------

    def on_page_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self.page_list.item(row)
        self.current_index = item.data(Qt.ItemDataRole.UserRole)
        self.refresh_preview()

    def refresh_preview(self) -> None:
        idx = self.current_index
        if idx < 0 or idx >= len(self.pages):
            return
        mode = self.preview_mode.currentIndex()
        if mode == 0 and idx in self.outputs:
            pix = QPixmap(str(self.outputs[idx]))
        elif mode == 2 and idx in self.inpainted:
            pix = QPixmap(str(self.inpainted[idx]))
        else:
            try:
                pix = to_pixmap(read_page_image(self.pages[idx]))
            except Exception as e:
                self.preview.setText(f'预览失败：{e}')
                return
        self.preview.setPixmap(pix)
        self.preview.resize(pix.size())

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, '选择输出目录', self.out_dir_edit.text())
        if folder:
            self.out_dir_edit.setText(folder)

    def open_output(self) -> None:
        path = Path(self.out_dir_edit.text())
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('MangaTransfer')
    # 显式指定中文字体，避免部分环境下默认字体缺字变成方块
    app.setFont(QFont('Microsoft YaHei UI', 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
