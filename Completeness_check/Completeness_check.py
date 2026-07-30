"""
整合区域选取 → 图像预处理+OCR识别 → 信息对比 的完整交互界面
【修正版】修复了步骤③对比逻辑的以下问题：
  1. 标签行解析：原正则无法匹配 "1:" 等格式 → 改为通用标签-行解析
  2. 匹配策略：原按页码累计数定位行 → 改为按图号范围+图名模式匹配
  3. 条目分组：原靠非空行计数循环 pos → 改为标签行+box1+box2 三行组解析
  4. box2 前导 "|" 处理：如 "|12-402"
  5. 兼容两种 TXT 格式：单页场景 "N:" 和 多页场景 "page_N:"
"""

import os
import sys
import json
import re
import shutil
import cv2
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QPlainTextEdit,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QMessageBox,
    QFrame, QSplitter
)
from PyQt5.QtGui import QFont, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer


# ============================================================
#  默认路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.join(BASE_DIR, 'images')
DEFAULT_REFPTS_PATH = os.path.join(BASE_DIR, 'refPts', 'refPts1.json')
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
DEFAULT_BOX_DIR = os.path.join(BASE_DIR, 'box_output')
DEFAULT_TXT_OUTPUT = os.path.join(BASE_DIR, 'output', 'recognition_result.txt')
DEFAULT_EXCEL_PATH = os.path.join(BASE_DIR, 'reference.xlsx')


# ============================================================
#  后台工作线程
# ============================================================
class Worker(QThread):
    """通用后台工作线程，避免 GUI 冻结"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str, object)  # success, message, result

    def __init__(self, target_func, *args, **kwargs):
        super().__init__()
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs
        self._result = None

    def run(self):
        try:
            # 重定向 print 到 log_signal
            import io
            from contextlib import redirect_stdout, redirect_stderr

            class EmittingStringIO(io.StringIO):
                def __init__(self, emitter):
                    super().__init__()
                    self.emitter = emitter

                def write(self, s):
                    super().write(s)
                    if s.strip():
                        self.emitter.emit(s.strip())

            buf = EmittingStringIO(self.log_signal)
            with redirect_stdout(buf), redirect_stderr(buf):
                self._result = self.target_func(*self.args, **self.kwargs)
            self.finished_signal.emit(True, "完成", self._result)
        except Exception as e:
            import traceback
            self.log_signal.emit(f"错误: {e}\n{traceback.format_exc()}")
            self.finished_signal.emit(False, str(e), None)



def load_refPts(path):
    """加载区域坐标 JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_images_in_folder(input_folder, output_folder):
    """二值化预处理所有图片"""
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_folder, filename))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, binary)
            print(f"预处理: {filename} -> {out_path}")


def recognize_text_and_save_boxes(img_path, refPts, output_folder, lang='chi_sim+eng'):
    """OCR 识别单张图片并保存 box 信息"""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'Completeness_check\Tesseract-OCR\tesseract.exe'

    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取: {img_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    basename = os.path.splitext(os.path.basename(img_path))[0]

    for i, (top_left, bottom_right) in enumerate(refPts):
        x1, y1 = top_left
        x2, y2 = bottom_right
        roi = img[y1:y2, x1:x2]

        # 保存裁剪图
        crop_path = os.path.join(output_folder, f'{basename}_crop_{i+1}.png')
        cv2.imwrite(crop_path, roi)

        # OCR 识别（--psm 6: 均匀文本块，更好保留间隔符号）
        boxes = pytesseract.image_to_boxes(roi, lang=lang, config='--psm 6')
        box_info = []
        for b in boxes.splitlines():
            parts = b.split(" ")
            if len(parts) == 6:
                try:
                    char, bx1, by1, bx2, by2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    h_roi = roi.shape[0]
                    box_info.append({
                        "character": char,
                        "bbox": [
                            x1 + bx1,
                            y1 + (h_roi - by1),
                            x1 + bx2,
                            y1 + (h_roi - by2),
                        ]
                    })
                except ValueError:
                    pass

        json_path = os.path.join(output_folder, f'{basename}_box_{i+1}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(box_info, f, ensure_ascii=False, indent=2)

    print(f"OCR 完成: {img_path}")


def process_folder(input_folder, refPts, output_folder):
    """批量处理文件夹中所有图片"""
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img_output = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(img_output, exist_ok=True)
            recognize_text_and_save_boxes(img_path, refPts, img_output)
            print(f"已完成: {filename}")


def process_json_files(folder_path, output_file_path):
    """合并所有 JSON box 文件为一个 TXT"""
    page_data = {}
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                box_label = 'box' + file_name.split('_box_')[-1].replace('.json', '')
                combined = ''.join(item['character'] for item in data)
                parent = os.path.basename(root)
                entry = f"{box_label}: {combined}\n"
                page_data[parent] = page_data.get(parent, '') + entry

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for parent in sorted(page_data.keys()):
            f.write(f"{parent}:\n{page_data[parent]}\n")
    print(f"结果已保存到: {output_file_path}")


# ---- 对比逻辑 (修正版) ----
def extract_chinese(text):
    return ''.join(re.findall(r'[一-鿿]', text))


def extract_numbers(text):
    return ''.join(re.findall(r'\d+', text))


def clean_number(text):
    return re.sub(r'\D', '', text)


def parse_number_range(excel_tuhao):
    """
    解析 Excel 图号范围。
    "12-401 ~ 404" → ("12-", 401, 404)
    "12-401"       → ("12-", 401, 401)
    "00-001 ~ 006" → ("00-", 1, 6)
    返回 (prefix, start_num, end_num) 或 None
    """
    tuhao = str(excel_tuhao).strip()
    m = re.match(r'([A-Za-z]*\d+[-_])?(\d+)\s*~\s*(\d+)', tuhao)
    if m:
        prefix = m.group(1) or ''
        return (prefix, int(m.group(2)), int(m.group(3)))
    m2 = re.match(r'([A-Za-z]*\d+[-_])(\d+)', tuhao)
    if m2:
        return (m2.group(1), int(m2.group(2)), int(m2.group(2)))
    return None


def number_in_range(extracted_tuhao, excel_tuhao):
    """判断提取的图号是否在 Excel 图号范围内"""
    extracted = clean_number(extracted_tuhao)
    range_info = parse_number_range(excel_tuhao)
    if range_info is None:
        return extracted == clean_number(excel_tuhao)
    prefix, start, end = range_info
    prefix_num = clean_number(prefix) if prefix else ''
    if prefix_num and extracted.startswith(prefix_num):
        num_part = extracted[len(prefix_num):]
    else:
        num_part = extracted
    if num_part and num_part.isdigit():
        return start <= int(num_part) <= end
    return False


def is_substring(sub, main):
    """放宽匹配：sub 中所有汉字是否都在 main 中出现"""
    sub_ch = extract_chinese(sub)
    main_ch = extract_chinese(main)
    if not sub_ch:
        return False
    return all(c in main_ch for c in sub_ch)


def find_matching_excel_row(extracted_tuhao, extracted_tuming, df):
    """
    按图号范围 + 图名在 Excel 中寻找最佳匹配行。
    返回 (row_series, tuming_match, tuhao_match)
    优先图号+图名均匹配的行，其次单项匹配。
    """
    best_row = None
    best_tm = False
    best_th = False
    best_score = -1
    for _, row in df.iterrows():
        th_match = number_in_range(extracted_tuhao, str(row['图号']))
        tm_match = is_substring(extracted_tuming or '', str(row.get('图名', '')))
        score = (1 if th_match else 0) + (1 if tm_match else 0)
        if score > best_score:
            best_row = row
            best_tm = tm_match
            best_th = th_match
            best_score = score
        if best_score == 2:
            break  # 完美匹配，立即返回
    return best_row, best_tm, best_th


def parse_txt_entries(text_lines):
    """
    通用 TXT 解析，支持两种格式：
    格式A（单页场景）: 1:\nbox1: ...\nbox2: ...
    格式B（多页场景）: page_1:\nbox1: ...\nbox2: ...
    返回 [{"label": str, "图名": str, "图号": str}, ...]
    """
    entries = []
    i = 0
    n = len(text_lines)
    while i < n:
        line = text_lines[i].strip()
        # 标签行: 以冒号结尾，且后续两行是 box 行
        if not line or not line.endswith(':'):
            i += 1
            continue
        if i + 2 >= n:
            break
        next1 = text_lines[i + 1].strip()
        next2 = text_lines[i + 2].strip()
        # 验证后续两行确实是 box1 / box2
        if not (re.match(r'box\d+:', next1) and re.match(r'box\d+:', next2)):
            i += 1
            continue

        label = line[:-1].strip()  # 去掉末尾冒号

        # 提取 box1（图名）
        box1_m = re.search(r'box\d+:\s*(.+)', next1)
        tuming = box1_m.group(1) if box1_m else ''

        # 提取 box2（图号），处理可能的前导 |
        box2_m = re.search(r'box\d+:\s*\|?\s*(.+)', next2)
        tuhao = box2_m.group(1) if box2_m else ''

        if tuming or tuhao:
            entries.append({"label": label, "图名": tuming, "图号": tuhao})
        i += 3  # 跳过已处理的三行
    return entries


def run_comparison(txt_path, excel_path):
    """执行完整对比并返回结果列表"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()

    df = pd.read_excel(excel_path)
    if '张数' in df.columns:
        df['累计张数'] = df['张数'].cumsum()
    else:
        df['累计张数'] = 0  # 防止不存在列时报错

    entries = parse_txt_entries(text_lines)
    print(f"从 TXT 解析到 {len(entries)} 个条目")

    results = []
    for entry in entries:
        row, tuming_match, tuhao_match = find_matching_excel_row(
            entry['图号'], entry['图名'], df
        )

        results.append({
            'page': entry['label'],
            'extracted_tuming': entry['图名'],
            'extracted_tuhao': entry['图号'],
            'excel_tuming': str(row['图名']) if row is not None else '未找到匹配行',
            'excel_tuhao': str(row['图号']) if row is not None else '未找到匹配行',
            'tuming_match': tuming_match,
            'tuhao_match': tuhao_match,
        })

    print(f"\n对比完成，共处理 {len(results)} 条记录")
    if results:
        matched = sum(1 for r in results if r['tuming_match'] and r['tuhao_match'])
        print(f"图名+图号完全匹配: {matched}/{len(results)}")
    return results


# ============================================================
#  OpenCV 区域选取窗口（独立运行，从 select.py 整合）
# ============================================================
def open_region_selector(image_path, output_path):
    """
    打开 OpenCV 交互窗口进行区域框选。
    操作：鼠标拖动画框 | r=重置 | q/ESC=保存并退出
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        QMessageBox.critical(None, "错误", f"无法读取图像: {image_path}")
        return []

    clone = img.copy()
    refPts = []
    currentPts = []
    cropping = False
    # 根据图片大小计算显示缩放：目标窗口约 1200x800，但原图比窗口小时不放大
    scale = min(1200 / img.shape[1], 800 / img.shape[0], 1.0)

    def on_mouse(event, x, y, flags, param):
        nonlocal cropping, currentPts
        x_orig = int(x / scale)
        y_orig = int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            currentPts = [(x_orig, y_orig)]
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            currentPts.append((x_orig, y_orig))
            cropping = False
            cv2.rectangle(img, currentPts[0], currentPts[1], (0, 255, 0), 2)
            refPts.append(tuple(currentPts))
            print(f"区域 {len(refPts)}: 左上{currentPts[0]} 右下{currentPts[1]}")

    cv2.namedWindow("区域选取 - Region Selector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("区域选取 - Region Selector",
                     int(img.shape[1] * scale), int(img.shape[0] * scale))
    cv2.setMouseCallback("区域选取 - Region Selector", on_mouse)

    # 在窗口内显示操作提示（OpenCV 仅支持英文/ASCII 字体）
    hint_lines = [
        "Drag to select region",
        "R=Reset  Q/ESC=Save & Exit",
        "Selected: 0",
    ]
    print("操作说明：鼠标拖动画框选取 | r=重置全部 | q/ESC=保存退出")

    while True:
        # 缩放显示：缩小用 INTER_AREA（清晰），放大用 INTER_CUBIC
        if scale < 1.0:
            display = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            display = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # 在图像顶部叠加操作提示文字
        hint_lines[2] = f"Selected: {len(refPts)}"
        y0 = 30
        for line in hint_lines:
            cv2.putText(display, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y0 += 28
        cv2.imshow("区域选取 - Region Selector", display)
        key = cv2.waitKey(1) & 0xFF

        try:
            if cv2.getWindowProperty("区域选取 - Region Selector", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if key == ord('r'):
            img = clone.copy()
            refPts.clear()
            print("已重置全部选区")
        elif key == ord('q') or key == 27:
            break

    cv2.destroyWindow("区域选取 - Region Selector")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refPts, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(refPts)} 个区域到: {output_path}")
    return refPts


# ============================================================
#  主 GUI
# ============================================================
class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图纸固定信息提取与对比系统")
        self.setGeometry(100, 50, 1100, 850)
        self.initUI()

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ---- 标题 ----
        title = QLabel("图纸固定信息提取与对比系统")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin: 10px 0;")
        main_layout.addWidget(title)

        # ---- 标签页 ----
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_step1_tab(), "① 区域选取")
        self.tabs.addTab(self._create_step2_tab(), "② 图像处理与OCR")
        self.tabs.addTab(self._create_step3_tab(), "③ 信息对比")
        main_layout.addWidget(self.tabs)

        # ---- 底部日志 ----
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(5000)
        self.log_text.setStyleSheet("font-family: Consolas, 'Microsoft YaHei'; font-size: 12px;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        self._log("系统就绪。请按 ①→②→③ 顺序操作。")

    # ========== 步骤 1：区域选取 ==========
    def _create_step1_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # 参考图片
        g1 = QGroupBox("参考图片（用于框选固定信息区域）")
        l1 = QHBoxLayout()
        self.step1_img_path = QLineEdit(os.path.join(DEFAULT_IMAGE_DIR, 'page_1.png'))
        l1.addWidget(QLabel("图片路径:"))
        l1.addWidget(self.step1_img_path)
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(lambda: self._browse_file(
            self.step1_img_path, "选择参考图片", "Images (*.png *.jpg *.jpeg)"))
        l1.addWidget(btn_browse)
        g1.setLayout(l1)

        # refPts 保存路径
        g2 = QGroupBox("区域坐标保存位置")
        l2 = QHBoxLayout()
        self.step1_refpts_path = QLineEdit(DEFAULT_REFPTS_PATH)
        l2.addWidget(QLabel("refPts:"))
        l2.addWidget(self.step1_refpts_path)
        btn_browse2 = QPushButton("浏览")
        btn_browse2.clicked.connect(lambda: self._browse_save(
            self.step1_refpts_path, "保存坐标文件", "JSON (*.json)"))
        l2.addWidget(btn_browse2)
        g2.setLayout(l2)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_open_selector = QPushButton("🖱 打开区域选取窗口")
        self.btn_open_selector.setStyleSheet("font-size: 14px; padding: 12px 30px; background-color: #0078d4; color: white;")
        self.btn_open_selector.clicked.connect(self._run_region_selector)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_open_selector)
        btn_layout.addStretch()

        layout.addWidget(g1)
        layout.addWidget(g2)
        layout.addLayout(btn_layout)
        layout.addStretch()

        tip = QLabel("提示：点击按钮后将打开 OpenCV 窗口，用鼠标拖动画框选取图名、图号等固定信息区域。")
        tip.setStyleSheet("color: gray; font-size: 12px;")
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)
        return w

    # ========== 步骤 2：图像处理与OCR ==========
    def _create_step2_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # 输入文件夹
        g1 = QGroupBox("输入：原始图片文件夹")
        l1 = QHBoxLayout()
        self.step2_input = QLineEdit(DEFAULT_IMAGE_DIR)
        l1.addWidget(self.step2_input)
        btn1 = QPushButton("浏览")
        btn1.clicked.connect(lambda: self._browse_dir(self.step2_input))
        l1.addWidget(btn1)
        g1.setLayout(l1)

        # refPts
        g2 = QGroupBox("区域坐标文件")
        l2 = QHBoxLayout()
        self.step2_refpts = QLineEdit(DEFAULT_REFPTS_PATH)
        l2.addWidget(self.step2_refpts)
        btn2 = QPushButton("浏览")
        btn2.clicked.connect(lambda: self._browse_file(
            self.step2_refpts, "选择坐标文件", "JSON (*.json)"))
        l2.addWidget(btn2)
        g2.setLayout(l2)

        # 预处理输出
        g3 = QGroupBox("预处理后图片输出")
        l3 = QHBoxLayout()
        self.step2_processed = QLineEdit(DEFAULT_PROCESSED_DIR)
        l3.addWidget(self.step2_processed)
        btn3 = QPushButton("浏览")
        btn3.clicked.connect(lambda: self._browse_dir(self.step2_processed))
        l3.addWidget(btn3)
        g3.setLayout(l3)

        # Box 输出
        g4 = QGroupBox("OCR 识别 Box 输出")
        l4 = QHBoxLayout()
        self.step2_box = QLineEdit(DEFAULT_BOX_DIR)
        l4.addWidget(self.step2_box)
        btn4 = QPushButton("浏览")
        btn4.clicked.connect(lambda: self._browse_dir(self.step2_box))
        l4.addWidget(btn4)
        g4.setLayout(l4)

        # TXT 输出
        g5 = QGroupBox("合并后 TXT 输出")
        l5 = QHBoxLayout()
        self.step2_txt = QLineEdit(DEFAULT_TXT_OUTPUT)
        l5.addWidget(self.step2_txt)
        btn5 = QPushButton("浏览")
        btn5.clicked.connect(lambda: self._browse_save(
            self.step2_txt, "保存 TXT", "Text (*.txt)"))
        l5.addWidget(btn5)
        g5.setLayout(l5)

        # 进度条 + 按钮
        btn_row = QHBoxLayout()
        self.step2_progress = QProgressBar()
        self.step2_progress.setVisible(False)
        self.btn_step2_run = QPushButton("▶ 开始预处理 + OCR 识别")
        self.btn_step2_run.setStyleSheet("font-size: 14px; padding: 12px 30px; background-color: #107c10; color: white;")
        self.btn_step2_run.clicked.connect(self._run_ocr_pipeline)
        btn_row.addWidget(self.step2_progress)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_step2_run)
        btn_row.addStretch()

        layout.addWidget(g1)
        layout.addWidget(g2)
        layout.addWidget(g3)
        layout.addWidget(g4)
        layout.addWidget(g5)
        layout.addLayout(btn_row)
        layout.addStretch()
        return w

    # ========== 步骤 3：信息对比 ==========
    def _create_step3_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # TXT 文件
        g1 = QGroupBox("OCR 识别结果 TXT")
        l1 = QHBoxLayout()
        self.step3_txt = QLineEdit(DEFAULT_TXT_OUTPUT)
        l1.addWidget(self.step3_txt)
        btn1 = QPushButton("浏览")
        btn1.clicked.connect(lambda: self._browse_file(
            self.step3_txt, "选择 TXT", "Text (*.txt)"))
        l1.addWidget(btn1)
        g1.setLayout(l1)

        # Excel 文件
        g2 = QGroupBox("参考 Excel 表格")
        l2 = QHBoxLayout()
        self.step3_excel = QLineEdit(DEFAULT_EXCEL_PATH)
        l2.addWidget(self.step3_excel)
        btn2 = QPushButton("浏览")
        btn2.clicked.connect(lambda: self._browse_file(
            self.step3_excel, "选择 Excel", "Excel (*.xlsx)"))
        l2.addWidget(btn2)
        g2.setLayout(l2)

        # 按钮
        btn_row = QHBoxLayout()
        self.btn_step3_run = QPushButton("▶ 开始对比")
        self.btn_step3_run.setStyleSheet("font-size: 14px; padding: 12px 30px; background-color: #d83b01; color: white;")
        self.btn_step3_run.clicked.connect(self._run_comparison)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_step3_run)
        btn_row.addStretch()

        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(8)
        self.result_table.setHorizontalHeaderLabels([
            "页码", "提取图名", "提取图号",
            "Excel图名", "Excel图号",
            "图名匹配", "图号匹配", "状态"
        ])
        self.result_table.setAlternatingRowColors(True)
        self.result_table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(g1)
        layout.addWidget(g2)
        layout.addLayout(btn_row)
        layout.addWidget(QLabel("对比结果（绿色=匹配，红色=不匹配）："))
        layout.addWidget(self.result_table)
        return w

    # ========== 文件浏览辅助 ==========
    def _browse_file(self, line_edit, title, filter_str):
        path = QFileDialog.getOpenFileName(self, title, line_edit.text(), filter_str)[0]
        if path:
            line_edit.setText(path)

    def _browse_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_save(self, line_edit, title, filter_str):
        path = QFileDialog.getSaveFileName(self, title, line_edit.text(), filter_str)[0]
        if path:
            line_edit.setText(path)

    def _log(self, msg):
        self.log_text.appendPlainText(msg)

    # ========== 步骤 1 执行 ==========
    def _run_region_selector(self):
        """在主线程中使用 QTimer 驱动 OpenCV 窗口，避免跨线程 GUI 异常"""
        # 安全检查：仅当 GUI 完全就绪后才允许打开区域选取
        if not getattr(self, '_gui_ready', False):
            self._log("⚠ GUI 尚未就绪，忽略区域选取请求")
            return

        img_path = self.step1_img_path.text()
        out_path = self.step1_refpts_path.text()

        if not os.path.isfile(img_path):
            QMessageBox.critical(self, "错误", f"图片不存在: {img_path}")
            return

        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            QMessageBox.critical(self, "错误", f"无法读取图像: {img_path}")
            return

        self._log(f"打开区域选取窗口: {img_path}")
        self._log("操作提示已显示在 OpenCV 窗口中（拖动画框 | R=重置 | Q/ESC=保存退出）")
        self.btn_open_selector.setEnabled(False)

        # 初始化 OpenCV 窗口（必须在主线程中创建）
        self._cv_img_path = img_path
        self._cv_out_path = out_path
        self._cv_img = img
        self._cv_clone = img.copy()
        self._cv_refPts = []
        self._cv_currentPts = []
        self._cv_cropping = False
        self._cv_scale = min(1200 / img.shape[1], 800 / img.shape[0], 1.0)
        self._cv_hint_lines = [
            "Drag to select region",
            "R=Reset  Q/ESC=Save & Exit",
            "Selected: 0",
        ]

        # 鼠标回调
        def on_mouse(event, x, y, flags, param):
            x_orig = int(x / self._cv_scale)
            y_orig = int(y / self._cv_scale)
            if event == cv2.EVENT_LBUTTONDOWN:
                self._cv_currentPts = [(x_orig, y_orig)]
                self._cv_cropping = True
            elif event == cv2.EVENT_LBUTTONUP:
                self._cv_currentPts.append((x_orig, y_orig))
                self._cv_cropping = False
                cv2.rectangle(self._cv_img, self._cv_currentPts[0], self._cv_currentPts[1], (0, 255, 0), 2)
                self._cv_refPts.append(tuple(self._cv_currentPts))
                self._log(f"区域 {len(self._cv_refPts)}: 左上{self._cv_currentPts[0]} 右下{self._cv_currentPts[1]}")

        cv2.namedWindow("区域选取 - Region Selector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("区域选取 - Region Selector",
                         int(img.shape[1] * self._cv_scale), int(img.shape[0] * self._cv_scale))
        cv2.setMouseCallback("区域选取 - Region Selector", on_mouse)

        # 使用 QTimer 在主线程中驱动 OpenCV 事件循环（不阻塞 Qt 事件处理）
        self._cv_timer = QTimer(self)
        self._cv_timer.timeout.connect(self._cv_tick)
        self._cv_timer.start(15)  # ~60fps

    def _cv_tick(self):
        """QTimer 回调：每帧刷新 OpenCV 窗口并处理按键"""
        try:
            # 检查窗口是否被用户关闭
            try:
                if cv2.getWindowProperty("区域选取 - Region Selector", cv2.WND_PROP_VISIBLE) < 1:
                    self._cv_finish()
                    return
            except cv2.error:
                self._cv_finish()
                return

            # 生成显示帧
            scale = self._cv_scale
            img = self._cv_img
            if scale < 1.0:
                display = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                display = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # 叠加提示文字
            self._cv_hint_lines[2] = f"Selected: {len(self._cv_refPts)}"
            y0 = 30
            for line in self._cv_hint_lines:
                cv2.putText(display, line, (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                y0 += 28

            cv2.imshow("区域选取 - Region Selector", display)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self._cv_img = self._cv_clone.copy()
                self._cv_refPts.clear()
                self._log("已重置全部选区")
            elif key == ord('q') or key == 27:  # ESC
                self._cv_finish()
        except Exception as e:
            import traceback
            self._log(f"OpenCV 窗口异常: {e}\n{traceback.format_exc()}")
            self._cv_cleanup()

    def _cv_finish(self):
        """正常结束 OpenCV 窗口并保存结果"""
        self._cv_timer.stop()
        self._cv_cleanup()

        # 保存区域坐标
        out_path = self._cv_out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self._cv_refPts, f, ensure_ascii=False, indent=2)
        self._log(f"已保存 {len(self._cv_refPts)} 个区域到: {out_path}")

        # 恢复按钮
        self.btn_open_selector.setEnabled(True)

    def _cv_cleanup(self):
        """销毁 OpenCV 窗口"""
        try:
            cv2.destroyWindow("区域选取 - Region Selector")
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass

    # ========== 步骤 2 执行 ==========
    def _run_ocr_pipeline(self):
        input_dir = self.step2_input.text()
        refpts_path = self.step2_refpts.text()
        processed_dir = self.step2_processed.text()
        box_dir = self.step2_box.text()
        txt_out = self.step2_txt.text()

        # 验证
        if not os.path.isdir(input_dir):
            QMessageBox.critical(self, "错误", "原始图片文件夹不存在")
            return
        if not os.path.isfile(refpts_path):
            QMessageBox.critical(self, "错误", "区域坐标文件不存在，请先在步骤①中选取")
            return

        self.btn_step2_run.setEnabled(False)
        self.step2_progress.setVisible(True)
        self.step2_progress.setRange(0, 0)

        def pipeline():
            self._log("--- 步骤② 开始 ---")
            refPts = load_refPts(refpts_path)
            self._log(f"加载坐标: {len(refPts)} 个区域")

            # 0/4 清理上次运行残留的输出目录，防止新旧结果混合
            for d in [processed_dir, box_dir]:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                    self._log(f"已清理旧输出: {d}")

            self._log("1/3 预处理图片...")
            preprocess_images_in_folder(input_dir, processed_dir)

            self._log("2/3 OCR 识别（使用原始图像以保证符号识别精度）...")
            process_folder(input_dir, refPts, box_dir)

            self._log("3/3 合并结果...")
            process_json_files(box_dir, txt_out)
            self._log("--- 步骤② 完成 ---")

        self._worker2 = Worker(pipeline)
        self._worker2.log_signal.connect(self._log)
        self._worker2.finished_signal.connect(self._on_step2_done)
        self._worker2.start()

    def _on_step2_done(self, success, msg, _result):
        self.btn_step2_run.setEnabled(True)
        self.step2_progress.setVisible(False)
        if success:
            QMessageBox.information(self, "完成", f"OCR 识别完成!\n结果: {self.step2_txt.text()}")
        else:
            QMessageBox.critical(self, "错误", msg)

    # ========== 步骤 3 执行 ==========
    def _run_comparison(self):
        txt_path = self.step3_txt.text()
        excel_path = self.step3_excel.text()

        if not os.path.isfile(txt_path):
            QMessageBox.critical(self, "错误", "TXT 文件不存在，请先完成步骤②")
            return
        if not os.path.isfile(excel_path):
            QMessageBox.critical(self, "错误", "Excel 文件不存在")
            return

        self.btn_step3_run.setEnabled(False)

        def do_compare():
            self._log("--- 步骤③ 开始对比 ---")
            return run_comparison(txt_path, excel_path)

        self._worker3 = Worker(do_compare)
        self._worker3.log_signal.connect(self._log)

        def on_finish(success, msg, results):
            self.btn_step3_run.setEnabled(True)
            if success and results:
                self._populate_table(results)
                QMessageBox.information(self, "完成", f"对比完成，共 {len(results)} 条")
            else:
                QMessageBox.critical(self, "错误", msg)

        self._worker3.finished_signal.connect(on_finish)
        self._worker3.start()

    def _populate_table(self, results):
        self.result_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.result_table.setItem(i, 0, QTableWidgetItem(r['page']))
            self.result_table.setItem(i, 1, QTableWidgetItem(r['extracted_tuming']))
            self.result_table.setItem(i, 2, QTableWidgetItem(r['extracted_tuhao']))
            self.result_table.setItem(i, 3, QTableWidgetItem(r['excel_tuming']))
            self.result_table.setItem(i, 4, QTableWidgetItem(r['excel_tuhao']))

            # 图名匹配
            item_tm = QTableWidgetItem("✓" if r['tuming_match'] else "✗")
            item_tm.setForeground(QColor("green") if r['tuming_match'] else QColor("red"))
            self.result_table.setItem(i, 5, item_tm)

            # 图号匹配
            item_th = QTableWidgetItem("✓" if r['tuhao_match'] else "✗")
            item_th.setForeground(QColor("green") if r['tuhao_match'] else QColor("red"))
            self.result_table.setItem(i, 6, item_th)

            # 综合状态
            both = r['tuming_match'] and r['tuhao_match']
            status = QTableWidgetItem("通过" if both else "不匹配")
            status.setForeground(QColor("green") if both else QColor("red"))
            self.result_table.setItem(i, 7, status)

        self.result_table.resizeColumnsToContents()


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 全局异常捕获
    def excepthook(exc_type, exc_value, exc_tb):
        import traceback
        err = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        QMessageBox.critical(None, "未捕获的异常", err[:2000])
        sys.exit(1)
    sys.excepthook = excepthook

    window = MainGUI()
    window._gui_ready = False  # 初始锁定，防止意外触发区域选取
    window.show()
    window._gui_ready = True   # GUI 显示后解锁
    sys.exit(app.exec_())
