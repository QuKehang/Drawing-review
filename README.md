# drawing-review — 桥梁工程设计工具集

图纸审查系统，集成 OCR、BERT 文本分类、YOLOv5 目标检测、RAG 知识库判别等功能。

## 项目结构

```
drawing-review/
├── pyproject.toml                          # uv 项目配置
├── launcher.py                             # 统一调度系统（一键启动）
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
├── PP-OCR_table_reading/                   # ④ 表格识别
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

### 基础环境（CPU 版本，所有机器通用）

默认安装 CPU 版 PaddlePaddle，适用于所有机器：

```bash
cd drawing-review
uv sync
```

### GPU 加速环境（需要 NVIDIA 显卡）

如果你有 NVIDIA 显卡（推荐 4 GB+ VRAM），可使用 GPU 版本获得 2–5 倍推理加速：

```bash
# 1. 用 GPU 配置文件覆盖默认配置
copy pyproject-gpu.toml pyproject.toml   # Windows
# cp pyproject-gpu.toml pyproject.toml    # macOS / Linux

# 2. 安装依赖（将自动安装 paddlepaddle-gpu）
uv sync

# 3. 安装 CUDA 运行库（cuDNN / cuBLAS / CUDA Runtime）
uv pip install "nvidia-cuda-runtime-cu11>=11.8.0,<12.0.0"
uv pip install "nvidia-cudnn-cu11>=8.9.0,<9.0.0"
uv pip install "nvidia-cublas-cu11>=11.0.0,<12.0.0"

# 4. 验证 GPU 是否可用
uv run python -c "import paddle; print('GPU 数量:', paddle.device.cuda.device_count())"
```

> **切换回 CPU 版本**：`git checkout pyproject.toml && uv sync`

> **PyPI 下载慢？** 可在 pip install 时使用清华镜像：
> `uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ nvidia-cuda-runtime-cu11 ...`

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

---

## 各模块操作步骤

### 统一调度系统（推荐）

项目提供了 PyQt5 图形化统一调度平台，可一键启动各子工具：

```bash
uv run python launcher.py
```

在调度界面中选择 Python 环境后，点击对应工具卡片的「▶ 启动」按钮即可运行。各工具以独立进程运行，关闭调度窗口不影响已启动的工具。

---

### ① YOLO 标注裁剪 — 目标检测 & 区域裁剪

**功能**：使用 YOLOv5 ONNX 模型检测图纸中的图名、标注、标题栏、绘图区域并自动裁剪分类保存。

**启动方式**：

```bash
uv run python Location/Drawing_location.py
```

**操作步骤**：

1. **选择输入目录** → 点击「选择文件夹」，选择包含待处理图纸图片（`.png` / `.jpg` / `.jpeg`）的文件夹
2. **选择输出目录** → 选择裁剪结果的保存位置
3. **预览图片** → 界面左侧将显示输入目录中的图片缩略图，可滚动浏览
4. **运行检测** → 点击「▶ 运行检测」，系统将使用 `cstr.onnx` 模型对每张图片进行目标检测
5. **查看结果** → 裁剪结果按标签分类保存，输出结构如下：

```
output_root/
├── figue/        # 图名区域
├── annotation/   # 标注/附注区域
├── title/        # 标题区域
├── title bar/    # 标题栏区域
└── draw/         # 绘图区域
```

**依赖**：YOLOv5 ONNX 模型（`Location/model/cstr.onnx`）+ PyQt5 GUI

---

### ② BERT 模型训练 — 设计说明合规性判别模型

**功能**：基于 `bert-base-chinese` 微调二分类模型（符合规范 / 不符合规范），为设计说明判定工具提供推理模型。

> ⚠️ **首次使用前必须运行此步骤**，训练完成后会在项目根目录生成 `bert_model_trained/` 文件夹。

**启动方式**：

```bash
uv run python Code_of_Bert_Trainning/Design_Judge.py
```

**操作步骤**：

1. **准备训练数据** → 确保 `Code_of_Bert_Trainning/` 目录下存在以下文件：
   - `design_spec.txt` — 正例训练数据（符合规范的设计说明条文）
   - `relation_no.txt` — 负例训练数据（不符合规范的文本示例）
2. **运行训练** → 执行脚本，将自动完成数据加载、模型微调、保存
3. **验证产出** → 确认 `bert_model_trained/` 目录下生成了 `config.json`、`model.safetensors`、`tokenizer.json` 等文件

**训练参数**（可在脚本中修改）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_LENGTH` | 128 | 最大序列长度 |
| `BATCH_SIZE` | 32 | 训练批次大小 |
| `EPOCHS` | 3 | 训练轮数 |
| `LEARNING_RATE` | 2e-5 | 学习率 |

---

### ③ 设计说明判定（BERT） — OCR + BERT 文本分类

**功能**：直接对已裁剪好的 annotation 图块文件夹进行 OCR 识别，使用本地 BERT 模型逐条判定每条附注是否符合设计说明规范。

**启动方式**：

```bash
uv run python Annotation-test/Annotation_check_without.py
```

**操作步骤**：

1. **准备裁剪图块** → 确保已有裁剪好的 annotation 区域图片（可由「YOLO 标注裁剪」工具生成）
2. **选择输入目录** → 点击「选择文件夹」，选择包含 annotation 裁剪图块的文件夹
3. **运行判定** → 点击「运行判别」，系统将：
   - 对每张裁剪图片进行预处理（对比度增强、中值滤波、二值化）
   - 使用 PaddleOCR 识别文字
   - 智能分句（支持全角/半角标点混合、连续编号项自动拆分）
   - 使用 BERT 模型逐条判定合规性
4. **查看结果** → 每条附注显示「符合设计说明」或「不符合设计说明」的分类结果

**前置条件**：必须先运行「BERT 模型训练」生成 `bert_model_trained/` 模型文件。

---

### ④ 基于 LLM 的设计说明判定 — DeepSeek-R1 + RAG

基于 **Ollama 本地部署的 DeepSeek-R1:8b** 模型，结合 **RAG（检索增强生成）** 知识库，对图纸附注进行智能合规性判别。相比纯 BERT 方案，能够给出详细的判断依据和引用的规范条文。

#### 安装 Ollama 并拉取模型

```bash
# 下载安装 Ollama: https://ollama.com

# 拉取推理模型（必选）
ollama pull deepseek-r1:8b

# 拉取嵌入模型（必选，用于向量搜索）
ollama pull nomic-embed-text
```

#### 启动方式

```bash
uv run python Annotation-test-RAG/Annotation_check_with.py
```

#### 操作步骤

1. **配置模型** → 确认 LLM 模型 (`deepseek-r1:8b`) 和 Embedding 模型 (`nomic-embed-text`) 已安装

2. **初始化知识库** → 将技术规范文档（txt/pdf/docx）放入 `user_docs/` 目录，点击「初始化知识库」

3. **添加文档** → 可随时点击「添加文档」追加新的规范文件到知识库

4. **选择文件夹** → 分别选择：
   - 裁剪图块文件夹（annotation 区域图片）
   - 输出结果保存文件夹

5. **运行判别** → 点击「▶ 运行判别」，系统会：
   - 对每张裁剪图片进行预处理和 OCR 识别
   - 智能分句拆分每条附注
   - 在知识库中检索相关技术规范
   - 由 DeepSeek-R1 综合判断并给出依据

6. **查看结果** → 可按「符合规范 / 不符合规范 / 无明确规定」分类筛选查看

7. **导出结果** → 点击「导出结果 (JSON)」保存完整判别记录

#### 判别结果格式

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

---

### ⑤ PP-OCR 表格识别 — 图纸表格结构化提取

**功能**：基于 PaddleOCR (PP-OCRv4 + PP-Structure) 对图纸中的表格进行结构化识别，支持两种模式。

**启动方式**：

```bash
uv run python PP-OCR_table_reading/Table_recognition.py
```

**操作步骤**：

1. **选择模式** → 在 GUI 界面中选择识别模式：
   - **JSON 标注模式**：根据 JSON 标注坐标裁剪表格区域后识别（适用于已标注的图纸）
   - **直接识别模式**：直接对整张图纸进行表格检测和识别
2. **选择目录** → 分别选择：
   - 输入图片/JSON 文件夹
   - 输出结果保存文件夹
3. **运行识别** → 点击运行，系统将：
   - 根据模式裁剪或检测表格区域
   - 使用 PP-Structure 进行表格结构化识别
4. **查看结果** → 结果保存为 Excel (`.xlsx`) + 文本 (`.txt`) 格式：

```
Output/
├── cropped_page_1_figue_2/
│   ├── [16, 20, 549, 624]_0.xlsx    # 表格数据
│   └── res_0.txt                    # 识别文本
└── ...
```

---

### ⑥ 固定信息提取与对比 — 完整性检查

**功能**：通过 OpenCV 交互式区域选取 → Tesseract OCR 识别 → 与 Excel 参考表自动比对，验证图纸的图名图号等信息完整性。

**启动方式**：

```bash
uv run python Completeness_check/Completeness_check.py
```

**操作步骤**：

1. **区域选取** → 在图纸图片上使用鼠标框选需要 OCR 识别的固定信息区域（如图名、图号位置）
2. **保存参考点** → 将选取的区域坐标保存为 `refPts/*.json` 参考点文件
3. **OCR 识别** → 系统使用 Tesseract 对选取区域进行文字识别，输出到 `output/recognition_result.txt`
4. **信息对比** → 加载 Excel 参考表（`reference.xlsx`），将 OCR 识别结果与参考表进行自动比对：
   - 按图号范围 + 图名模式匹配
   - 支持单页 (`N:`) 和多页 (`page_N:`) 两种标注格式
   - 自动标记匹配/不匹配项
5. **查看结果** → 在 GUI 的对比表格中查看匹配状态，定位不一致的条目

**前置条件**：需安装 Tesseract OCR 并将安装目录置于 `Completeness_check/` 下（见上方 Tesseract OCR 安装说明）。

---

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

## 硬件要求

### 基础配置（BERT / YOLO / OCR / 表格识别）

适用于不使用 RAG + DeepSeek 的场景：

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 x86-64 | 8 核以上 |
| 内存 | 8 GB | 16 GB |
| 显卡 | 无（CPU 推理即可） | NVIDIA GPU 4 GB+ VRAM（加速 BERT 训练与 PaddleOCR 推理） |
| 磁盘空间 | 5 GB（含依赖与模型） | 10 GB |

> **关于 GPU**：PaddleOCR (PP-OCRv4) 模型本身为轻量级设计，CPU 即可流畅运行，GPU 主要加速批量处理场景（约 2–5 倍）。少量图纸处理 CPU 完全足够。
>
> **启用 GPU 加速**：如需 GPU 推理，请使用 `pyproject-gpu.toml` 覆盖默认配置并安装 NVIDIA 运行库，详见上方「GPU 加速环境」章节。

### 完整配置（含 RAG + DeepSeek-R1:8b）

使用 `Annotation_check_with.py`（DeepSeek-R1 + RAG 判别）需满足以下额外要求：

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 8 核 x86-64（支持 AVX2） | 12 核以上 |
| 内存 | 16 GB | 32 GB |
| 显卡 | NVIDIA GPU 8 GB+ VRAM（4-bit 量化，速度提升显著） | NVIDIA GPU 12 GB+ VRAM |
| 磁盘空间 | 20 GB（含 Ollama 模型约 5 GB） | 30 GB+（含知识库文档与输出结果） |

> **说明**：
> - **DeepSeek-R1:8b** 是硬件需求最大的组件：4-bit 量化约需 4.7 GB 存储 / 6 GB 内存，若使用 CPU 推理（无 GPU）需有足够内存带宽，处理速度会明显慢于 GPU。
> - **nomic-embed-text** 嵌入模型约需 274 MB 存储 / 1 GB 内存。
> - 如仅使用 BERT 判别方案（`Annotation_check_without.py`），无需 GPU，8 GB 内存即可流畅运行。

### 各模型资源占用参考

| 模型 | 存储占用 | 运行时内存 | 说明 |
|------|----------|------------|------|
| YOLOv5 ONNX (`cstr.onnx`) | 28 MB | ~500 MB | CPU / GPU 均可 |
| BERT (`bert-base-chinese`) | ~400 MB | ~1 GB | 训练需额外 2-4 GB |
| PaddleOCR (PP-OCRv4) | ~100 MB | ~1 GB | 首次运行自动下载 |
| Tesseract OCR (chi_sim) | ~15 MB | ~200 MB | 需手动安装 |
| DeepSeek-R1:8b (4-bit) | ~4.7 GB | ~6 GB | Ollama 管理 |
| nomic-embed-text | ~274 MB | ~1 GB | Ollama 管理 |

## 注意事项

1. **首次使用**：需先运行 `Design_Judge.py` 训练 BERT 模型，生成 `bert_model_trained/` 后才能使用设计说明判定功能。
2. **双硬件配置**：项目默认使用 CPU 版 PaddlePaddle（所有机器通用）。如需 GPU 加速，将 `pyproject-gpu.toml` 覆盖 `pyproject.toml` 后重新 `uv sync`，详见环境配置章节。`Table_recognition.py` 启动时会自动检测 GPU 可用性并切换推理设备。
3. **OpenCV GUI**：`uv sync` 后 `opencv-python-headless` 可能被自动安装，运行 `uv run python scripts/fix_opencv.py` 即可修复。
4. **模型路径**：YOLOv5 模型（`cstr.onnx`）和推理脚本位于 `Location/model/` 目录下。
5. **逐条检验**：设计说明判定已优化分句逻辑，支持全角/半角标点混合、连续编号项自动拆分，确保每条附注独立判定。
6. **Tesseract 路径**：完整性检查模块依赖 Tesseract OCR，需单独安装并将安装目录置于 `Completeness_check/` 下。
