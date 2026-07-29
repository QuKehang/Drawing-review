"""
直接对已裁剪好的 annotation 图块文件夹做 OCR + BERT 判错。
不需要 JSON 坐标文件，不需要原始大图 — 图片已经是裁剪好的 annotation 区域。

"""

import torch  # 必须在 PaddleOCR 之前导入，否则 albumentations→torch 的 DLL 加载会失败
import re
import sys
import os
import numpy as np
from paddleocr import PaddleOCR
from transformers import BertTokenizer, BertForSequenceClassification
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFileDialog, QMainWindow, QPlainTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageEnhance, ImageFilter
from PyQt5.QtCore import Qt


# ============================================================
#  加载 BERT 模型
# ============================================================
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'bert_model_trained')
)

labels = ['不符合设计说明', '符合设计说明']
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# PaddleOCR 全局实例（延迟初始化，避免重复加载模型）
_ocr_instance = None


def get_ocr():
    """获取 PaddleOCR 单例"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='ch',
                                  show_log=False, use_gpu=False)
    return _ocr_instance


# 默认 annotation 文件夹
DEFAULT_ANNOTATION_DIR = r'\Annotation-test\input'


# ============================================================
#  辅助函数 (与原 Judge.py 相同)
# ============================================================
def predict_sentence(sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = labels[torch.argmax(logits)]
    return predicted_label


def correct_ocr_errors(text):
    text = text.replace('Z', '/')
    return text


def replace_colon_with_semicolon(text):
    result = re.sub(r'(?<=:)(?=\s*[0-9]{1,2})', ';', text)
    return result


def split_and_number(text):
    pattern = r'(?<=;)|(?<=。)'
    split_strings = re.split(pattern, text)
    split_strings = [s.strip().lstrip('0123456789.') for s in split_strings if s.strip()]
    numbered_strings = [f"{i + 1}.{s}" for i, s in enumerate(split_strings)]
    return numbered_strings


def preprocess_image(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.filter(ImageFilter.MedianFilter())
    img = img.convert('L')
    threshold = 128
    img = img.point(lambda p: p > threshold and 255)
    return img


# ============================================================
#  核心逻辑：直接对裁剪好的 annotation 图块做 OCR + BERT
# ============================================================
def judge_crops(annotation_img_folder):
    """
    直接读取文件夹中已裁剪好的图片，
    逐张 OCR → 逐句 BERT 判断是否符合设计说明。
    返回: {文件名: [(句子, 判断结果), ...]}
    """
    all_results = {}

    image_files = [f for f in os.listdir(annotation_img_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        return {"(空)": [("未找到任何图片文件", "")]}

    for img_file in sorted(image_files):
        img_path = os.path.join(annotation_img_folder, img_file)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            all_results[img_file] = [(f"(读取失败: {e})", "")]
            continue

        # 图像预处理
        preprocessed_img = preprocess_image(img)

        # OCR 识别（PaddleOCR）
        ocr = get_ocr()
        img_array = np.array(preprocessed_img)
        ocr_result = ocr.ocr(img_array, cls=True)
        if ocr_result and ocr_result[0]:
            text = ''.join([line[1][0] for line in ocr_result[0]])
        else:
            text = ''
        text = correct_ocr_errors(text)
        text = replace_colon_with_semicolon(text)

        # 清理空白
        text = " ".join([line.strip() for line in text.splitlines() if line.strip()])
        text = text.replace(" ", "")

        if not text:
            all_results[img_file] = [("(OCR未识别到文字)", "")]
            continue

        # 按分号/句号分割为多条
        text_list = split_and_number(text)

        # 逐句 BERT 判错
        list_result = []
        for sentence in text_list:
            predicted_label = predict_sentence(sentence)
            list_result.append((sentence, predicted_label))

        all_results[img_file] = list_result

    return all_results


# ============================================================
#  PyQt5 GUI
# ============================================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_index = 0
        self.image_files = []
        self.initUI()

    def initUI(self):
        self.setGeometry(200, 100, 1000, 800)
        self.setWindowTitle('Annotation 图块判错 — 直接处理裁剪图片')

        # ---- 控件 ----
        self.annotation_folder_path = QLineEdit(self)
        self.annotation_folder_path.setText(DEFAULT_ANNOTATION_DIR)
        self.result_text = QPlainTextEdit(self)
        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet('background-color: #f0f0f0; border: 1px solid #ccc;')

        self.load_img_button = QPushButton('📁 选择 annotation 图片文件夹', self)
        self.prev_button = QPushButton('◀ 上一张', self)
        self.next_button = QPushButton('下一张 ▶', self)
        self.run_button = QPushButton('▶ 运行 OCR + BERT 判断', self)
        self.exit_button = QPushButton('退出程序', self)

        self.run_button.setStyleSheet('font-size: 14px; padding: 8px;')

        # ---- 信号连接 ----
        self.load_img_button.clicked.connect(self.load_annotation_folder)
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.run_button.clicked.connect(self.run_judge)
        self.exit_button.clicked.connect(QApplication.quit)

        # ---- 布局 ----
        nav_layout = QVBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.annotation_folder_path)
        layout.addWidget(self.load_img_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.img_label)
        layout.addLayout(nav_layout)
        layout.addWidget(self.result_text)
        layout.addWidget(self.exit_button)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 启动时加载默认路径
        self.refresh_image_list()

    def refresh_image_list(self):
        folder = self.annotation_folder_path.text()
        if folder and os.path.isdir(folder):
            self.image_files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.current_image_index = 0
            if self.image_files:
                self.show_image_at_index(0)
            else:
                self.img_label.clear()
                self.img_label.setText('(文件夹中没有图片)')

    def load_annotation_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 annotation 裁剪图片文件夹")
        if folder:
            self.annotation_folder_path.setText(folder)
            self.refresh_image_list()

    def show_image_at_index(self, index):
        folder = self.annotation_folder_path.text()
        if 0 <= index < len(self.image_files):
            img_file = os.path.join(folder, self.image_files[index])
            image = QImage(img_file)
            if image.isNull():
                return
            max_width = 800
            max_height = 600
            self.img_label.setMaximumSize(max_width, max_height)
            scaled_image = image.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(QPixmap(scaled_image))
            self.img_label.setScaledContents(True)
            self.img_label.setToolTip(f'{index + 1}/{len(self.image_files)}: {self.image_files[index]}')

    def show_prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image_at_index(self.current_image_index)

    def show_next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image_at_index(self.current_image_index)

    def run_judge(self):
        annotation_path = self.annotation_folder_path.text()
        self.result_text.clear()

        if not annotation_path or not os.path.isdir(annotation_path):
            self.result_text.setPlainText('请先选择有效的 annotation 图片文件夹。')
            return

        self.result_text.setPlainText('正在处理中，请稍候...')
        QApplication.processEvents()

        result = judge_crops(annotation_path)

        lines = []
        for img_file, judgements in result.items():
            lines.append(f'{"=" * 60}')
            lines.append(f'📷 {img_file}')
            lines.append(f'{"=" * 60}')
            for sentence, label in judgements:
                lines.append(f'  {sentence} —— {label}')
            lines.append('')

        self.result_text.setPlainText('\n'.join(lines))

    def closeEvent(self, event):
        self.img_label.clear()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Application closed.")
