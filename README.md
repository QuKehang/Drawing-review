# drawing-review — 桥梁工程设计工具集

图纸审查系统，集成 OCR、BERT 文本分类、YOLOv5 目标检测、RAG 知识库判别等功能。

## 项目结构

```
drawing-review/
├── pyproject.toml                          # uv 项目配置
├── launcher_v2.py                          # 统一调度系统（一键启动）
│
├── Annotation-test/                        # ① 设计说明判定（BERT）
│   └── Annotation_check_without.py         #    OCR + BERT 分类，直接处理裁剪图块
│
├── Annotation-test-RAG/                    # ② 设计说明判定（RAG + DeepSeek）
│   ├── Annotation_check_with.py            #    OCR + RAG + DeepSeek-R1 判别
│   └── RAG.py                              #    本地知识库系统（ChromaDB + Ollama）
│
├── Location/                               # ③ 目标检测 & 区域裁剪
│   ├── yolov5_partition.py                 #    YOLOv5 ONNX 推理引擎
│   └── Drawing_location.py                 #    按标签分类裁剪 + GUI
│
├── PP-OCR _table_reading/                  # ④ 表格识别
│   └── Table_recognition.py                #    PP-OCRv4 表格结构化识别
│
├── Code_of_Bert_Trainning/                 # ⑤ BERT 模型训练
│   └── Design_Judge.py                     #    bert-base-chinese 微调二分类
│
└── Completeness_check/                     # ⑥ 完整性检查
    └── Completeness_check.py               #    区域选取 + OCR + Excel 信息对比
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

## 运行方式

```bash
# 统一调度系统（推荐）
uv run python launcher_v2.py

# 或单独运行各工具
uv run python Annotation-test/Annotation_check_without.py
uv run python Annotation-test-RAG/Annotation_check_with.py
uv run python PP-OCR _table_reading/Table_recognition.py
uv run python Location/Drawing_location.py
uv run python Code_of_Bert_Trainning/Design_Judge.py
uv run python Completeness_check/Completeness_check.py
```

## 工具说明

| 工具 | 功能 | 核心技术 |
|------|------|---------|
| 设计说明判定 | OCR 提取图纸附注 → 自动判别合规性 | PaddleOCR + BERT / DeepSeek-R1 |
| 表格识别 | 图纸表格结构化提取 → Excel/HTML | PP-OCRv4 + PP-Structure |
| 目标检测裁剪 | 检测图纸区域 → 按标签分类裁剪 | YOLOv5 ONNX |
| BERT 训练 | 微调 bert-base-chinese 二分类模型 | Transformers |
| 完整性检查 | 区域选取 → OCR → Excel 信息对比 | Tesseract + PyQt5 |

## 依赖

- Python >= 3.10, < 3.13
- PyTorch, Transformers
- PaddleOCR, PaddlePaddle
- OpenCV, Pillow
- PyQt5
- 可选：LangChain, ChromaDB, Ollama（RAG 判别）
