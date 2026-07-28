"""
crop_by_label.py
采用与 main.py 相同的 YOLOv5 ONNX 模型 (cstr.onnx)，
对输入图片进行目标检测，将各标签对应的图块从原图中裁剪下来，
并按标签名分类保存到对应的子文件夹中。

输出结构示例:
    output_root/
    ├── figue/
    │   ├── page_577_crop_0.png
    │   └── ...
    ├── annotation/
    │   └── ...
    ├── title/
    ├── title bar/
    └── draw/

运行方式:
    python crop_by_label.py
"""

import sys
import os

# 将 Annotation-test 目录加入搜索路径，以便导入 yolov5_partition
ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'Annotation-test')
ANNOTATION_DIR = os.path.abspath(ANNOTATION_DIR)
sys.path.insert(0, ANNOTATION_DIR)

import json
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMessageBox, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QWidget, QScrollArea, QFormLayout, QProgressBar, QGroupBox
)
from PIL import Image

from yolov5_partition import yolov5

# 模型文件路径
MODEL_PATH = os.path.join(ANNOTATION_DIR, 'cstr.onnx')


class ImageScroller(QWidget):
    """可滚动展示输入目录中图片缩略图的组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)

        self.image_layout = QVBoxLayout()
        self.image_container = QWidget()
        self.image_container.setLayout(self.image_layout)

        self.scroll_area.setWidget(self.image_container)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

    def show_images(self, directory):
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        if not os.path.isdir(directory):
            return

        files = os.listdir(directory)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
                image.thumbnail((800, 500))
                data = image.tobytes('raw', 'RGB')
                qimage = QtGui.QImage(data, image.width, image.height,
                                      image.width * 3, QtGui.QImage.Format_RGB888)
                label = QLabel()
                label.setPixmap(QtGui.QPixmap.fromImage(qimage))
                label.setToolTip(image_file)
                self.image_layout.addWidget(label)
            except Exception as e:
                print(f'[警告] 无法加载缩略图 {image_file}: {e}')


class CropApp(QWidget):
    """主界面：选择目录 → 检测并裁剪 → 按标签名分文件夹保存"""

    def __init__(self):
        super().__init__()
        self.input_directory = ''
        self.output_root_directory = ''
        self.init_ui()

    # ---------------------------------------------------------------
    #  UI 搭建
    # ---------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle('图块裁剪 — 按标签分类保存')
        self.setMinimumSize(800, 600)

        # ---- 输入区 ----
        input_group = QGroupBox('输入设置')
        input_layout = QVBoxLayout()

        row1 = QVBoxLayout()
        self.input_button = QPushButton('📁 选择输入图片目录')
        self.input_button.clicked.connect(self.browse_input_directory)
        self.input_label = QLabel('未选择')
        self.input_label.setStyleSheet('color: gray;')
        row1.addWidget(self.input_button)
        row1.addWidget(self.input_label)
        input_layout.addLayout(row1)

        row2 = QVBoxLayout()
        self.output_button = QPushButton('📂 选择裁剪输出根目录')
        self.output_button.clicked.connect(self.browse_output_directory)
        self.output_label = QLabel('未选择')
        self.output_label.setStyleSheet('color: gray;')
        row2.addWidget(self.output_button)
        row2.addWidget(self.output_label)
        input_layout.addLayout(row2)

        input_group.setLayout(input_layout)

        # ---- 运行区 ----
        run_group = QGroupBox('执行')
        run_layout = QVBoxLayout()
        self.run_button = QPushButton('▶  开始检测并裁剪')
        self.run_button.setStyleSheet('font-size: 14px; padding: 10px;')
        self.run_button.clicked.connect(self.run_crop)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: gray;')
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.progress_bar)
        run_layout.addWidget(self.status_label)
        run_group.setLayout(run_layout)

        # ---- 图片预览 ----
        self.image_scroller = ImageScroller()

        # ---- 主布局 ----
        main_layout = QVBoxLayout()
        main_layout.addWidget(input_group)
        main_layout.addWidget(run_group)
        main_layout.addWidget(QLabel('输入图片预览:'))
        main_layout.addWidget(self.image_scroller)
        self.setLayout(main_layout)

    # ---------------------------------------------------------------
    #  目录选择
    # ---------------------------------------------------------------
    def browse_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, '选择输入图片目录')
        if directory:
            self.input_directory = directory
            self.input_label.setText(directory)
            self.input_label.setStyleSheet('color: green;')
            self.image_scroller.show_images(directory)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, '选择输出根目录')
        if directory:
            self.output_root_directory = directory
            self.output_label.setText(directory)
            self.output_label.setStyleSheet('color: green;')

    # ---------------------------------------------------------------
    #  核心逻辑：检测 + 裁剪 + 分类保存
    # ---------------------------------------------------------------
    def run_crop(self):
        if not self.input_directory:
            QMessageBox.critical(self, '错误', '请先选择输入图片目录。')
            return
        if not self.output_root_directory:
            QMessageBox.critical(self, '错误', '请先选择输出根目录。')
            return
        if not os.path.exists(MODEL_PATH):
            QMessageBox.critical(self, '错误',
                                 f'模型文件未找到:\n{MODEL_PATH}\n\n'
                                 f'请确认 cstr.onnx 位于 Annotation-test 目录下。')
            return

        try:
            self.run_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText('正在初始化 YOLOv5 模型...')
            QtWidgets.QApplication.processEvents()

            # ---- 初始化模型（与 main.py 使用相同的 cstr.onnx 权重） ----
            model = yolov5(MODEL_PATH)

            # ---- 收集所有图片 ----
            image_files = []
            for root, dirs, files in os.walk(self.input_directory):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append((root, f))

            total = len(image_files)
            if total == 0:
                QMessageBox.warning(self, '警告',
                                    '输入目录中没有图片文件 (.png / .jpg / .jpeg)。')
                self.run_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                return

            self.progress_bar.setMaximum(total)
            crop_count = {}  # 统计各类别裁剪数量
            error_list = []

            # ---- 逐张处理 ----
            for idx, (root, filename) in enumerate(image_files):
                self.progress_bar.setValue(idx)
                self.status_label.setText(f'处理中 ({idx + 1}/{total}): {filename}')
                QtWidgets.QApplication.processEvents()

                image_path = os.path.join(root, filename)
                srcimg = cv2.imread(image_path)
                if srcimg is None:
                    error_list.append(f'无法读取: {filename}')
                    continue

                # 运行检测（与 main.py 相同的 detect 方法）
                _ = model.detect(srcimg, filename)

                # 收集当前图片的检测结果（用于 JSON 输出）
                detection_results = []

                # 遍历所有检测结果
                for box in model.figures_boxes:
                    # 提取纯标签名 ("figue: 0.70" → "figue")
                    label_raw = box['label']
                    label_name = label_raw.split(':')[0].strip()
                    conf = float(label_raw.split(':')[1].strip()) if ':' in label_raw else None

                    coords = box['coordinates']
                    x1 = max(0, coords['left'])
                    y1 = max(0, coords['top'])
                    x2 = coords['right']
                    y2 = coords['bottom']

                    # 记录检测结果
                    detection_results.append({
                        'label': label_raw,
                        'label_name': label_name,
                        'confidence': conf,
                        'coordinates': {
                            'left': x1,
                            'top': y1,
                            'right': x2,
                            'bottom': y2
                        }
                    })

                    # 裁剪图块
                    crop = srcimg[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # 创建标签名对应的子文件夹
                    label_dir = os.path.join(self.output_root_directory, label_name)
                    os.makedirs(label_dir, exist_ok=True)

                    # 生成唯一文件名
                    base_name = os.path.splitext(filename)[0]
                    idx_in_label = crop_count.get(label_name, 0)
                    crop_filename = f'{base_name}_crop_{idx_in_label}.png'
                    crop_path = os.path.join(label_dir, crop_filename)

                    cv2.imwrite(crop_path, crop)
                    crop_count[label_name] = idx_in_label + 1

                # ---- 保存 JSON 检测结果 ----
                json_dir = os.path.join(self.output_root_directory, 'jsons')
                os.makedirs(json_dir, exist_ok=True)
                base_name = os.path.splitext(filename)[0]
                json_path = os.path.join(json_dir, f'{base_name}.json')
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(detection_results, jf, ensure_ascii=False, indent=2)

            # ---- 完成 ----
            self.progress_bar.setValue(total)

            summary_parts = []
            for k in sorted(crop_count.keys()):
                summary_parts.append(f'{k}: {crop_count[k]} 个')
            summary = '\n'.join(summary_parts) if summary_parts else '未检测到任何标签图块'

            if error_list:
                summary += f'\n\n⚠ 以下文件处理失败:\n' + '\n'.join(error_list[:10])

            self.status_label.setText(f'完成！共处理 {total} 张图片')
            QMessageBox.information(
                self, '处理完成',
                f'共处理 {total} 张图片\n\n'
                f'裁剪统计:\n{summary}\n\n'
                f'JSON 检测结果: {os.path.join(self.output_root_directory, "jsons")}\n'
                f'输出目录: {self.output_root_directory}'
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, '运行错误', f'{e}')
        finally:
            self.run_button.setEnabled(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CropApp()
    window.setGeometry(200, 100, 900, 750)
    window.show()
    sys.exit(app.exec_())
