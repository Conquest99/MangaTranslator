# 模型资源

此目录保存模型清单和检查工具，不保存模型权重。BallonsTranslator 的模型加载代码使用相对于其项目根目录的 `data/models` 路径，因此权重仍放在：

```text
BallonsTranslator/data/models/
BallonsTranslator/data/alphabet-all-v7.txt
```

主流程 GUI 默认使用 CTD 文本检测、PaddleOCR-VL Manga OCR 和 LaMa 修复。GUI 还提供 `mit48px`、`manga_ocr`、`ppv6_onnx` OCR 选项；清单记录了这些选项依赖的文件和上游下载地址。

检查默认 GUI 选项所需文件：

```powershell
python 整合\models\check_models.py
```

检查 CPU 检测或其他 OCR 配置：

```powershell
python 整合\models\check_models.py --device cpu
python 整合\models\check_models.py --ocr-engine mit48px
python 整合\models\check_models.py --ocr-engine manga_ocr
python 整合\models\check_models.py --ocr-engine ppv6_onnx
```

检查器只验证文件是否存在，不会下载权重。缺失文件时会打印对应上游地址；也可以通过 BallonsTranslator 的模型下载流程获取。模型权重通常很大（默认 PaddleOCR-VL 权重大约 1.9 GB），且需遵守各模型仓库的许可条款，所以不纳入本仓库的普通 Git 提交。Sakura 本地翻译模型也仍由用户单独准备；DeepSeek API 密钥不能提交。
