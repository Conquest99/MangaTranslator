# 个人用机翻

漫画图片 / CBZ 的本地批量翻译工具。支持文本检测、OCR、机器翻译、原文修复和译文嵌字。

## 运行依赖

本项目复用 [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) 的检测、OCR、翻译和修复模块。代码从相邻目录加载它，因此请把两个项目放在同一个父目录：

```text
workspace/
├─ MangaTranslator/       # 本项目
└─ BallonsTranslator/     # 上游依赖
```

在 `workspace` 下克隆上游项目，并按照它的说明准备 Python 环境、依赖和模型：

```powershell
git clone https://github.com/dmMaze/BallonsTranslator.git
```

模型文件由 BallonsTranslator 加载，默认放在 `BallonsTranslator/data/models/`。本项目提供模型清单和检查工具：

```powershell
python MangaTranslator\models\check_models.py
```

## 启动

使用已安装本项目依赖和 BallonsTranslator 依赖的 Python 环境，在项目目录运行：

```powershell
python app.py
```

GUI 支持图片、文件夹和 CBZ。默认输出目录为 `outputs/gui/`，也可以在界面中更改。

## 翻译器

- **DeepSeek API**：需要在环境变量 `DEEPSEEK_API_KEY` 中配置密钥，或将密钥写入 `.secrets/deepseek.key`。密钥文件不会提交到 Git。
- **Copy Source**：用于不调用外部服务的流程测试。
- **Sakura**：仅保留程序支持；Sakura 模型和本地推理服务文件不随本仓库发布。使用前需自行准备模型并启动兼容服务。

## 目录与隐私

- `models/manifest.json` 列出默认 GUI 和可选 OCR 配置需要的模型文件及其上游下载地址。
- 模型权重不存入本仓库；请遵守各模型仓库的许可条款。
- `data/`、`outputs/`、`.secrets/`、`Sakura-7B/` 及本地 OCR 标注文件均由 `.gitignore` 排除。
- `samples/sample_page.png` 是由 `samples/make_sample_page.py` 生成的合成测试页。

## 单步脚本与评测

- `引入引擎.py`：文本检测
- `引入ORC.py`：OCR
- `引入翻译.py`：翻译
- `擦除.py`：原文修复
- `渲染嵌字.py`：译文排版渲染
- `ocr_对比.py`、`批量评测.py`：OCR 对比与批量评测

`pipeline.py` 是 GUI 和批处理主流程共用的后端。
