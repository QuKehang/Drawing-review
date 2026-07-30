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
安装时勾选中文语言包 (Chinese Simplified)

Tesseract OCR下载也可通过阿里云盘下载，位于云盘的drawing-review\OCR文件夹，链接为https://www.alipan.com/s/NshsKZPU32Z
下载后安装路径都应置于Drawing_review文件夹下

### YOLO预训练模型下载
预训练模型文件格式为onnx，保存在阿里云盘的drawing-review\Model Profile文件夹，链接为https://www.alipan.com/s/NshsKZPU32Z

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

## 设计说明判定系统 —— DeepSeek-R1 + RAG

基于 **Ollama 本地部署的 DeepSeek-R1:8b** 模型，结合 **RAG（检索增强生成）** 知识库，
对工程图纸中的附注（annotation）内容进行规范性判别，并**给出具体的判别依据**。

### 功能特点

- 🔍 **RAG 知识库**：支持用户自行添加 **txt / pdf / docx** 格式的技术规范文档
- 🤖 **本地推理**：通过 Ollama 调用 DeepSeek-R1:8b，数据不出本机
- 📝 **判据透明**：每条判别结果包含：判断结果 + 判断依据 + 参考规范条文 + 检索来源
- 🖼️ **OCR 集成**：自动对图纸附注区域进行 OCR 文字识别后逐条判别
- 🎛️ **PyQt5 桌面界面**：完整的知识库管理 + 文件选择 + 结果筛选 + JSON 导出

### 安装 Ollama 并拉取模型

```bash
# 下载安装 Ollama: https://ollama.com

# 拉取推理模型（必选）
ollama pull deepseek-r1:8b

# 拉取嵌入模型（必选，用于向量搜索）
ollama pull nomic-embed-text
```

### 操作步骤

1. **配置模型** → 确认 LLM 模型 (`deepseek-r1:8b`) 和 Embedding 模型 (`nomic-embed-text`) 已安装

2. **初始化知识库** → 将技术规范文档（txt/pdf/docx）放入 `user_docs/` 目录，点击「初始化知识库」

3. **添加文档** → 可随时点击「添加文档」追加新的规范文件到知识库

4. **选择文件** → 选择包含图纸图片的文件夹和对应的 JSON 标注文件夹

5. **运行判别** → 点击「▶ 运行判别」，系统会：
   - 读取 JSON 中的 annotation 坐标
   - 裁切图片区域并 OCR 识别文字
   - 对每条附注在知识库中检索相关规范
   - 由 DeepSeek-R1 综合判断并给出依据

6. **查看结果** → 可按「符合规范/不符合规范/无明确规定」筛选查看

7. **导出结果** → 点击「导出结果 (JSON)」保存完整判别记录

## 判别结果格式

每条判别输出包含：

```
─────────────────────────────────────────────────
文件: page_577.json  |  附注 #1
附注原文: 1.注:桥梁设计使用年限为100年
判断结果: 【符合规范】
判断依据: 该附注要求的设计使用年限为100年，与本规范第1.0.4条...
相关规范条文: 1.0.4 公路钢混组合桥梁...应按不小于100年设计使用年限...
检索来源: 公路钢混组合桥梁设计与施工规范.txt(p.?), ...
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
