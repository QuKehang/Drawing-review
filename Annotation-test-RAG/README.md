# 附注判别系统 —— DeepSeek-R1 + RAG

基于 **Ollama 本地部署的 DeepSeek-R1:8b** 模型，结合 **RAG（检索增强生成）** 知识库，
对工程图纸中的附注（annotation）内容进行规范性判别，并**给出具体的判别依据**。

## 功能特点

- 🔍 **RAG 知识库**：支持用户自行添加 **txt / pdf / docx** 格式的技术规范文档
- 🤖 **本地推理**：通过 Ollama 调用 DeepSeek-R1:8b，数据不出本机
- 📝 **判据透明**：每条判别结果包含：判断结果 + 判断依据 + 参考规范条文 + 检索来源
- 🖼️ **OCR 集成**：自动对图纸附注区域进行 OCR 文字识别后逐条判别
- 🎛️ **PyQt5 桌面界面**：完整的知识库管理 + 文件选择 + 结果筛选 + JSON 导出

## 环境准备

### 1. 安装 Ollama 并拉取模型

```bash
# 下载安装 Ollama: https://ollama.com

# 拉取推理模型（必选）
ollama pull deepseek-r1:8b

# 拉取嵌入模型（必选，用于向量搜索）
ollama pull nomic-embed-text
```

### 2. 安装 Tesseract OCR

下载安装: https://github.com/UB-Mannheim/tesseract/wiki
安装时勾选中文语言包 (Chinese Simplified)

如果安装路径不同，修改 `Judge.py` 中的：
```python
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseracy-OCR\tesseract.exe'
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动程序

```bash
python Judge.py
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

## 目录结构

```
Annotation-test-RAG/
├── Judge.py              # 主程序（PyQt5 桌面应用）
├── RAG.py                # RAG 知识库模块
├── requirements.txt      # Python 依赖
├── README.md             # 本文档
├── user_docs/            # 用户文档目录（放 txt/pdf/docx）
├── 规范文件/              # 示例规范文件
└── chroma_db/            # 向量数据库（自动生成）
```
