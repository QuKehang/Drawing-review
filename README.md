# drawing-review — 桥梁工程设计工具集

图纸审查系统，集成 OCR、BERT 文本分类、YOLOv5 目标检测、RAG 知识库判别等功能。

## 项目结构

```
drawing-review/
├── pyproject.toml                          # uv 项目配置
├── launcher_v2.py                          # 统一调度系统（一键启动）
├── uv.lock                                 # 依赖锁定文件
│
├── Annotation-test/                        # ① 设计说明判定（BERT）
│   └── Annotation_check_without.py         #    OCR + BERT 分类，直接处理裁剪图块
│
├── Annotation-test-RAG/                    # ② 设计说明判定（RAG + DeepSeek）
│   ├── Annotation_check_with.py            #    OCR + RAG + DeepSeek-R1 判别
│   └── RAG.py                              #    本地知识库系统（ChromaDB + Ollama）
│
├── Location/                               # ③ 目标检测 & 区域裁剪
│   ├── Drawing_location.py                 #    按标签分类裁剪 + GUI
│   └── model/                              #    YOLOv5 模型文件
│       ├── cstr.onnx                       #    ONNX 权重
│       └── yolov5_partition.py             #    YOLOv5 推理引擎
│
├── PP-OCR _table_reading/                  # ④ 表格识别
│   └── Table_recognition.py                #    PP-OCRv4 表格结构化识别
│
├── Code_of_Bert_Trainning/                 # ⑤ BERT 模型训练
│   ├── Design_Judge.py                     #    bert-base-chinese 微调二分类
│   ├── design_spec.txt                     #    正例训练数据（符合规范）
│   └── relation_no.txt                     #    负例训练数据（不符合规范）
│
├── bert_model_trained/                     # ⑥ BERT 训练产出（git-ignored）
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── Completeness_check/                     # ⑦ 完整性检查
│   └── Completeness_check.py               #    区域选取 + OCR + Excel 信息对比
│
└── scripts/
    └── fix_opencv.py                       #    OpenCV GUI 修复脚本
```

## 环境配置

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

### 安装 uv

```bash
pip install uv
```

### 基础环境（OCR / BERT / YOLO / 表格识别）

```bash
cd drawing-review
uv sync
```

### 完整环境（含 RAG 知识库 + DeepSeek 判别）

```bash
uv sync --extra rag
```
> 仅 `Annotation_check_with.py` 和 `RAG.py` 需要 RAG 额外依赖（LangChain、ChromaDB、Ollama 等）。


### 国内网络适配

Hugging Face 模型下载可能受限，项目已在脚本中自动配置 `HF_ENDPOINT=https://hf-mirror.com` 镜像站。

### 检验配置环境（OpenCV GUI 修复）
`albumentations → albucore` 传递依赖 `opencv-python-headless`，该包与 `opencv-python` 的 GUI 功能冲突，会导致 `cv2.namedWindow` / `cv2.imshow` 不可用。`uv sync` 后运行：

```bash
uv run python scripts/fix_opencv.py
```

### Tesseract OCR安装
下载地址`https://github.com/UB-Mannheim/tesseract/wiki`
Tesseract OCR下载也可通过阿里云盘下载，链接为https://www.alipan.com/s/NshsKZPU32Z，位于drawing-review\OCR文件夹中
下载后安装路径都应置于Drawing_review\Completeness_check文件夹下

### YOLO预训练模型下载
预训练模型文件格式为onnx，保存在阿里云盘中，链接为https://www.alipan.com/s/NshsKZPU32Z，位于其中drawing-review\Model Profile文件夹中

## 运行方式

### 统一调度系统（推荐）

```bash
uv run python launcher.py
```

### 单独运行各工具

```bash
# BERT 模型训练（首次使用前必须运行一次）
uv run python Code_of_Bert_Trainning/Design_Judge.py

# 设计说明判定 — 直接处理裁剪图块（BERT）
uv run python Annotation-test/Annotation_check_without.py

# 设计说明判定 — RAG + DeepSeek 判别
uv run python Annotation-test-RAG/Annotation_check_with.py

# 表格识别
uv run python PP-OCR_table_reading/Table_recognition.py

# 目标检测裁剪 — 按标签分类保存
uv run python Location/Drawing_location.py

# 完整性检查 — 区域选取 + OCR + Excel 对比
uv run python Completeness_check/Completeness_check.py
```

## 工具说明

| 工具 | 功能 | 核心技术 | 关键文件 |
|------|------|---------|----------|
| BERT 训练 | 微调 bert-base-chinese 二分类模型 | Transformers + PyTorch | `Design_Judge.py` |
| 设计说明判定 | OCR 提取图纸附注 → 逐条判定合规性 | PaddleOCR + BERT / DeepSeek-R1 | `Annotation_check_without.py` |
| 表格识别 | 图纸表格结构化提取 → Excel/HTML | PP-OCRv4 + PP-Structure | `Table_recognition.py` |
| 目标检测裁剪 | 检测图纸区域 → 按标签分类裁剪 | YOLOv5 ONNX + PyQt5 | `Drawing_location.py` |
| 完整性检查 | 区域选取 → OCR → Excel 信息对比 | Tesseract + PyQt5 + OpenCV | `Completeness_check.py` |

## 核心技术栈

| 类别 | 技术 |
|------|------|
| OCR | PaddleOCR、Tesseract |
| 文本分类 | BERT (bert-base-chinese)、DeepSeek-R1 |
| 目标检测 | YOLOv5 ONNX (cstr.onnx) |
| 表格识别 | PP-OCRv4 + PP-Structure |
| RAG | LangChain + ChromaDB + Ollama |
| 图像处理 | OpenCV、Pillow |
| GUI | PyQt5 |
| 数据处理 | Pandas、openpyxl |

## 依赖

- Python >= 3.10, < 3.13
- PyTorch >= 2.0
- Transformers >= 4.30
- PaddleOCR >= 2.9, PaddlePaddle >= 2.6
- OpenCV >= 4.8, Pillow >= 10.0
- PyQt5 >= 5.15
- 可选：LangChain、ChromaDB、Ollama（RAG 判别）

## 注意事项

1. **首次使用**：需先运行 `Design_Judge.py` 训练 BERT 模型，生成 `bert_model_trained/` 后才能使用设计说明判定功能。
2. **OpenCV GUI**：`uv sync` 后 `opencv-python-headless` 可能被自动安装，运行 `uv run python scripts/fix_opencv.py` 即可修复。
3. **模型路径**：YOLOv5 模型（`cstr.onnx`）和推理脚本位于 `Location/model/` 目录下。
4. **逐条检验**：设计说明判定已优化分句逻辑，支持全角/半角标点混合、连续编号项自动拆分，确保每条附注独立判定。
