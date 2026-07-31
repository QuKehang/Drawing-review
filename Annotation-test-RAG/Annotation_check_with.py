import sys
import os
import re
import json

# ── OCR ──────────────────────────────────────────────────
import numpy as np
import torch  # 必须在 PaddleOCR 之前导入，否则 albumentations→torch 的 DLL 加载会失败
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter

# PaddleOCR 全局单
_ocr_instance = None


def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='ch',
                                  show_log=False, use_gpu=False)
    return _ocr_instance

# ── GUI ──────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QPlainTextEdit, QGroupBox, QFormLayout, QProgressBar,
    QMessageBox, QListWidget, QAbstractItemView, QSplitter,
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QTextCursor, QColor, QTextCharFormat
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ── RAG ──────────────────────────────────────────────────
from RAG import LocalKnowledgeBase


# ═══════════════════════════════════════════════════════════
#  OCR 工具函数
# ═══════════════════════════════════════════════════════════

def correct_ocr_errors(text: str) -> str:
    return text.replace('Z', '/')


def replace_colon_with_semicolon(text: str) -> str:
    return re.sub(r'(?<=:)(?=\s*[0-9]{1,2})', ';', text)


def split_and_number(text: str) -> list:
    # 按中文/英文分号、中文句号、换行符分割（均在分隔符之后断开）
    pattern = r'(?<=[;；。\n])'
    parts = re.split(pattern, text)
    parts = [s.strip().lstrip('0123456789.；;，,') for s in parts if s.strip()]
    return [f"{i + 1}.{s}" for i, s in enumerate(parts)]


def preprocess_image(img: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.filter(ImageFilter.MedianFilter())
    img = img.convert('L')
    img = img.point(lambda p: p > 128 and 255)
    return img


def ocr_image(img: Image.Image) -> list:
    """对单张裁剪图片做预处理 + OCR → 返回句子列表"""
    preprocessed = preprocess_image(img)
    img_array = np.array(preprocessed)
    ocr_result = get_ocr().ocr(img_array, cls=True)
    if ocr_result and ocr_result[0]:
        text = ''.join([line[1][0] for line in ocr_result[0]])
    else:
        text = ''
    text = correct_ocr_errors(text)
    text = replace_colon_with_semicolon(text)
    text = text.replace(" ", "")
    if not text:
        return []
    return split_and_number(text)


# ═══════════════════════════════════════════════════════════
#  判别后台线程
# ═══════════════════════════════════════════════════════════

class JudgeWorker(QThread):
    progress = pyqtSignal(int, int)       # current, total
    status = pyqtSignal(str)              # 状态消息
    result_ready = pyqtSignal(dict)       # 单条结果
    finished = pyqtSignal(list)           # 全部结果 [{img_file, sentences: [...]}, ...]
    error = pyqtSignal(str)

    def __init__(self, kb: LocalKnowledgeBase, image_folder: str):
        super().__init__()
        self.kb = kb
        self.image_folder = image_folder

    def run(self):
        all_results = []
        try:
            image_files = sorted([
                f for f in os.listdir(self.image_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if not image_files:
                self.error.emit("未找到图片文件 (png / jpg / jpeg)")
                return

            total = len(image_files)
            self.status.emit(f"共 {total} 张裁剪图片待处理")

            for idx, img_file in enumerate(image_files):
                self.status.emit(f"[{idx + 1}/{total}] 正在处理: {img_file}")
                img_path = os.path.join(self.image_folder, img_file)

                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    all_results.append({"image": img_file, "error": str(e), "sentences": []})
                    continue

                sentences = ocr_image(img)
                if not sentences:
                    all_results.append({"image": img_file, "error": "OCR 未识别到文字", "sentences": []})
                    self.progress.emit(idx + 1, total)
                    continue

                judged_sentences = []
                for sent_idx, sentence in enumerate(sentences):
                    self.status.emit(
                        f"[{idx + 1}/{total}] {img_file} → "
                        f"条目 {sent_idx + 1}/{len(sentences)}"
                    )
                    try:
                        jr = self.kb.judge_annotation(sentence, k=4)
                    except Exception as e:
                        jr = {
                            "annotation": sentence,
                            "result": "判别失败",
                            "basis": str(e),
                            "clause": "",
                            "sources": [],
                        }
                    judged_sentences.append(jr)
                    self.result_ready.emit({"image": img_file, **jr})

                all_results.append({
                    "image": img_file,
                    "error": None,
                    "sentences": judged_sentences,
                })
                self.progress.emit(idx + 1, total)

            self.finished.emit(all_results)

        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════

class JudgeCropApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.kb: LocalKnowledgeBase = None
        self.worker: JudgeWorker = None
        self.all_results = []              # [{image, sentences: [...]}]
        self.all_sentences = []            # [{image, ...}]  flat list for filtering
        self.image_files = []              # 图片文件名列表
        self.current_image_index = 0
        self.init_ui()

    # ── UI 初始化 ────────────────────────────────────────

    def init_ui(self):
        self.setGeometry(150, 80, 1300, 900)
        self.setWindowTitle('附注判别系统 —— DeepSeek-R1 + RAG（裁剪图版）')

        central = QWidget()
        self.setCentralWidget(central)

        # 主水平分割：左侧图像 + 右侧结果
        main_splitter = QSplitter(Qt.Horizontal)

        # ── 左侧面板：图像 + 控制 ─────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # 模型配置
        cfg_group = QGroupBox("模型配置")
        cfg_form = QFormLayout()
        self.model_edit = QLineEdit("deepseek-r1:8b")
        self.embed_edit = QLineEdit("nomic-embed-text")
        cfg_form.addRow("LLM:", self.model_edit)
        cfg_form.addRow("Embedding:", self.embed_edit)
        cfg_group.setLayout(cfg_form)
        left_layout.addWidget(cfg_group)

        # 知识库管理
        kb_group = QGroupBox("知识库（txt / pdf / docx）")
        kb_layout = QVBoxLayout()

        kb_btn_row = QHBoxLayout()
        self.btn_init_kb = QPushButton("初始化知识库")
        self.btn_init_kb.clicked.connect(self.init_kb)
        self.btn_add_docs = QPushButton("添加文档")
        self.btn_add_docs.clicked.connect(self.add_documents)
        self.btn_rebuild = QPushButton("清空重建")
        self.btn_rebuild.clicked.connect(self.rebuild_kb)
        kb_btn_row.addWidget(self.btn_init_kb)
        kb_btn_row.addWidget(self.btn_add_docs)
        kb_btn_row.addWidget(self.btn_rebuild)
        kb_layout.addLayout(kb_btn_row)

        self.kb_status_label = QLabel("状态：未初始化")
        self.kb_status_label.setStyleSheet("color: gray; font-weight: bold;")
        kb_layout.addWidget(self.kb_status_label)
        kb_group.setLayout(kb_layout)
        left_layout.addWidget(kb_group)

        # 图片文件夹选择
        img_group = QGroupBox("裁剪图片文件夹")
        img_sel_layout = QHBoxLayout()
        self.img_folder_edit = QLineEdit()
        self.btn_img = QPushButton("浏览...")
        self.btn_img.clicked.connect(self.select_image_folder)
        img_sel_layout.addWidget(self.img_folder_edit)
        img_sel_layout.addWidget(self.btn_img)
        img_group.setLayout(img_sel_layout)
        left_layout.addWidget(img_group)

        # 运行 / 停止
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶ 运行判别")
        self.btn_run.clicked.connect(self.run_judge)
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; font-size: 14px; padding: 8px 24px; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_judge)
        self.btn_stop.setEnabled(False)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_stop)
        left_layout.addLayout(run_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪 — 请先初始化知识库，然后选择裁剪图片文件夹")
        self.status_label.setStyleSheet("color: #555; font-style: italic;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # 图像显示
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(400, 300)
        self.img_label.setStyleSheet(
            'background-color: #2d2d2d; border: 1px solid #555;'
        )
        self.img_label.setText('(裁剪图片预览)')
        left_layout.addWidget(self.img_label, 1)

        # 图片导航
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton('◀ 上一张')
        self.btn_next = QPushButton('下一张 ▶')
        self.image_info_label = QLabel('0 / 0')
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.btn_prev.clicked.connect(self.show_prev_image)
        self.btn_next.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.image_info_label)
        nav_layout.addWidget(self.btn_next)
        left_layout.addLayout(nav_layout)

        # ── 右侧面板：结果 ──────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        result_group = QGroupBox("判别结果")
        result_layout = QVBoxLayout()

        # 筛选按钮
        filter_layout = QHBoxLayout()
        self.btn_show_all = QPushButton("全部")
        self.btn_show_pass = QPushButton("符合规范")
        self.btn_show_fail = QPushButton("不符合规范")
        self.btn_show_unknown = QPushButton("无明确规定")
        self.btn_show_all.clicked.connect(lambda: self.filter_results("全部"))
        self.btn_show_pass.clicked.connect(lambda: self.filter_results("符合规范"))
        self.btn_show_fail.clicked.connect(lambda: self.filter_results("不符合规范"))
        self.btn_show_unknown.clicked.connect(lambda: self.filter_results("规范中无明确规定"))
        for b in [self.btn_show_all, self.btn_show_pass, self.btn_show_fail, self.btn_show_unknown]:
            filter_layout.addWidget(b)
        result_layout.addLayout(filter_layout)

        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Microsoft YaHei", 10))
        self.result_text.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; }"
        )
        result_layout.addWidget(self.result_text, 1)

        btn_export = QPushButton("导出结果 (JSON)")
        btn_export.clicked.connect(self.export_results)
        result_layout.addWidget(btn_export)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        # ── 组装 splitter ──────────────────────────────────
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)

        outer_layout = QVBoxLayout(central)
        outer_layout.addWidget(main_splitter)

    # ── 知识库管理 ────────────────────────────────────────

    def _ensure_kb(self):
        """如果 KB 未初始化则自动初始化（懒加载），返回 True 表示成功"""
        if self.kb is not None:
            return True
        try:
            model = self.model_edit.text().strip() or "deepseek-r1:8b"
            embed = self.embed_edit.text().strip() or "nomic-embed-text"
            docs_dir = "./user_docs"
            persist_dir = "./chroma_db"

            self.kb = LocalKnowledgeBase(
                model_name=model, embedding_model=embed,
                persist_dir=persist_dir,
            )
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                self.kb.load_existing_kb()
            else:
                os.makedirs(docs_dir, exist_ok=True)
                self.kb.build_knowledge_base(docs_dir)  # 空目录也能初始化

            stats = self.kb.get_stats()
            self.kb_status_label.setText(f"状态：已就绪 | 向量块数: {stats['chunks']}")
            self.kb_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.btn_run.setEnabled(True)
            self.btn_add_docs.setEnabled(True)
            self.btn_rebuild.setEnabled(True)
            return True
        except Exception as e:
            self.kb_status_label.setText(f"状态：失败 - {str(e)[:80]}")
            self.kb_status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(self, "初始化失败", str(e))
            return False

    def init_kb(self):
        """手动初始化 / 重新加载知识库"""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.kb_status_label.setText("状态：正在初始化...")
        self.kb_status_label.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()
        self.kb = None  # 强制重建
        self._ensure_kb()
        QApplication.restoreOverrideCursor()

    def add_documents(self):
        """添加文档 —— 若 KB 未初始化则自动初始化"""
        if not self._ensure_kb():
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文档（可多选）",
            filter="文档文件 (*.txt *.pdf *.docx);;所有文件 (*.*)",
        )
        if not files:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.kb_status_label.setText("状态：正在添加文档...")
            self.kb_status_label.setStyleSheet("color: orange; font-weight: bold;")
            QApplication.processEvents()

            total_added = 0
            failed_files = []
            for f in files:
                try:
                    n = self.kb.add_documents(f)
                    total_added += n
                except Exception as file_err:
                    failed_files.append((os.path.basename(f), str(file_err)))

            stats = self.kb.get_stats()
            if failed_files:
                # 部分文件失败
                fail_detail = "\n".join(
                    f"• {name}: {err[:120]}" for name, err in failed_files
                )
                self.kb_status_label.setText(
                    f"状态：部分成功 ({len(files) - len(failed_files)}/{len(files)} 个文件) | 向量块数: {stats['chunks']}")
                self.kb_status_label.setStyleSheet("color: orange; font-weight: bold;")
                QMessageBox.warning(
                    self, "添加完成（部分失败）",
                    f"成功添加 {len(files) - len(failed_files)}/{len(files)} 个文件，"
                    f"{total_added} 个文本块\n\n"
                    f"以下文件添加失败:\n{fail_detail}"
                )
            else:
                self.kb_status_label.setText(
                    f"状态：已添加 {len(files)} 个文件 | 向量块数: {stats['chunks']}")
                self.kb_status_label.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(self, "完成",
                                        f"成功添加 {len(files)} 个文件，{total_added} 个文本块")
        except Exception as e:
            QMessageBox.warning(self, "添加失败", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def rebuild_kb(self):
        reply = QMessageBox.question(
            self, "确认", "是否清空知识库并重新构建？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.kb_status_label.setText("状态：正在重建...")
            self.kb_status_label.setStyleSheet("color: orange; font-weight: bold;")
            QApplication.processEvents()

            import shutil, gc, time

            # 1. 先释放旧的 ChromaDB 连接
            if self.kb is not None:
                try:
                    self.kb.vector_store._client.delete_collection("knowledge_base")
                except Exception:
                    pass
                try:
                    self.kb.vector_store._client.clear_system_cache()
                except Exception:
                    pass
                self.kb = None
                gc.collect()

            # 2. 删除旧的向量库文件（含重试）
            persist_dir = os.path.abspath("./chroma_db")
            for attempt in range(5):
                try:
                    if os.path.exists(persist_dir):
                        shutil.rmtree(persist_dir)
                    break
                except PermissionError:
                    time.sleep(0.5)
                    gc.collect()

            # 3. 重建
            os.makedirs("./user_docs", exist_ok=True)
            self.kb = LocalKnowledgeBase(
                model_name=self.model_edit.text().strip() or "deepseek-r1:8b",
                embedding_model=self.embed_edit.text().strip() or "nomic-embed-text",
                persist_dir=persist_dir,
            )
            self.kb.build_knowledge_base("./user_docs")
            stats = self.kb.get_stats()
            self.kb_status_label.setText(f"状态：已重建 | 向量块数: {stats['chunks']}")
            self.kb_status_label.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, "完成", "知识库已重建")
        except Exception as e:
            self.kb_status_label.setText(f"状态：重建失败")
            self.kb_status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.warning(self, "重建失败", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    # ── 图片文件夹选择 & 导航 ────────────────────────────

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择裁剪图片文件夹")
        if folder:
            self.img_folder_edit.setText(folder)
            self.refresh_image_list()

    def refresh_image_list(self):
        folder = self.img_folder_edit.text()
        if folder and os.path.isdir(folder):
            self.image_files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.current_image_index = 0
            if self.image_files:
                self.show_image_at_index(0)
            else:
                self.img_label.setText('(文件夹中没有图片)')

    def show_image_at_index(self, index):
        folder = self.img_folder_edit.text()
        if 0 <= index < len(self.image_files):
            img_path = os.path.join(folder, self.image_files[index])
            image = QImage(img_path)
            if image.isNull():
                return
            scaled = image.scaled(700, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(QPixmap(scaled))
            self.img_label.setToolTip(
                f'{index + 1}/{len(self.image_files)}: {self.image_files[index]}'
            )
            self.image_info_label.setText(
                f'{index + 1} / {len(self.image_files)}\n{self.image_files[index]}'
            )

    def show_prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image_at_index(self.current_image_index)

    def show_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image_at_index(self.current_image_index)

    # ── 判别运行 ──────────────────────────────────────────

    def run_judge(self):
        if not self._ensure_kb():
            return
        img_folder = self.img_folder_edit.text().strip()
        if not img_folder:
            QMessageBox.warning(self, "提示", "请先选择裁剪图片文件夹")
            return

        self.all_results = []
        self.all_sentences = []
        self.result_text.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = JudgeWorker(self.kb, img_folder)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.on_status)
        self.worker.result_ready.connect(self.on_result)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_judge(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(2000)
            self.status_label.setText("已停止")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def on_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_status(self, msg: str):
        self.status_label.setText(msg)

    def on_result(self, result: dict):
        self.all_sentences.append(result)

    def on_finished(self, results: list):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.all_results = results

        if not results:
            self.status_label.setText("未找到任何图片或附注内容")
            return

        total_sentences = sum(len(r["sentences"]) for r in results)
        pass_count = sum(
            1 for r in results
            for s in r["sentences"]
            if s.get("result", "") == "符合规范"
        )
        fail_count = sum(
            1 for r in results
            for s in r["sentences"]
            if s.get("result", "") == "不符合规范"
        )
        unknown_count = sum(
            1 for r in results
            for s in r["sentences"]
            if s.get("result", "") == "规范中无明确规定"
        )
        error_count = sum(1 for r in results if r.get("error"))

        self.status_label.setText(
            f"完成 | {len(results)} 张图片, {total_sentences} 条附注 | "
            f"符合: {pass_count} | 不符合: {fail_count} | "
            f"无明确规定: {unknown_count}"
            + (f" | 识别失败: {error_count}" if error_count else "")
        )

        # 展示全部结果
        self.filter_results("全部")

    def on_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "错误", msg)

    # ── 结果展示 ──────────────────────────────────────────

    RESULT_COLORS = {
        "符合规范": QColor("#4CAF50"),
        "不符合规范": QColor("#f44336"),
        "规范中无明确规定": QColor("#FF9800"),
        "判别失败": QColor("#9e9e9e"),
    }

    def _append_result_to_widget(self, r: dict):
        """以富文本追加单条判别结果"""
        cursor = self.result_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        def insert(text, color=None, bold=False, italic=False, size=10):
            fmt = QTextCharFormat()
            fmt.setForeground(color or QColor("#cccccc"))
            fmt.setFontPointSize(size)
            fmt.setFontWeight(QFont.Bold if bold else QFont.Normal)
            fmt.setFontItalic(italic)
            cursor.insertText(text, fmt)

        result_type = r.get("result", "未知")
        color = self.RESULT_COLORS.get(result_type, QColor("#cccccc"))

        insert(f"\n{'─' * 80}\n", QColor("#555555"))
        insert(f"📷 {r.get('image', '?')}\n", QColor("#64B5F6"), bold=True, size=11)
        insert(f"附注原文: {r.get('annotation', '')}\n", QColor("#ffffff"), bold=True)
        insert(f"判断结果: 【{result_type}】\n", color, bold=True)

        basis = r.get("basis", "")
        if basis and basis != "无":
            insert(f"判断依据: {basis}\n", QColor("#aaaaaa"))

        clause = r.get("clause", "")
        if clause:
            insert(f"相关规范条文: {clause}\n", QColor("#81C784"), italic=True)

        sources = r.get("sources", [])
        if sources:
            src_strs = [f"{s['source']}(p.{s.get('page','?')})" for s in sources[:3]]
            insert(f"检索来源: {', '.join(src_strs)}\n", QColor("#555555"))

        self.result_text.setTextCursor(cursor)
        self.result_text.ensureCursorVisible()

    def filter_results(self, filter_type: str):
        self.result_text.clear()
        if filter_type == "全部":
            show = self.all_sentences
        else:
            show = [r for r in self.all_sentences
                    if r.get("result", "") == filter_type]
        for r in show:
            self._append_result_to_widget(r)

        # 也更新结果汇总信息
        if self.all_results:
            total_s = sum(len(r["sentences"]) for r in self.all_results)
        else:
            total_s = 0
        self.status_label.setText(
            f"筛选: {filter_type} — 显示 {len(show)}/{total_s} 条")

    def export_results(self):
        if not self.all_results:
            QMessageBox.information(self, "提示", "没有可导出的结果")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "judge_results.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "导出成功", f"已保存至: {path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(2000)
        event.accept()


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = JudgeCropApp()
    window.show()
    sys.exit(app.exec_())
