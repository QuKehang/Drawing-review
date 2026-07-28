"""
tesseract.py — OCR 识别模块（改进版）
核心改进：
  1. 小图自动上采样（Tesseract 在 300+ DPI 等效分辨率下效果最佳）
  2. 图像锐化 + 自适应对比度增强
  3. 使用 image_to_data 获取置信度，过滤低置信度字符
  4. 后处理：去除孤立低置信字符、修正常见混淆
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import json
import os
import re

# ---- Tesseract 路径 ----
PYTESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_and_print_refPts(refPts_path):
    """读取 JSON 文件并打印选择的区域。"""
    with open(refPts_path, 'r') as f:
        refPts = json.load(f)
    for i, pts in enumerate(refPts):
        print(f"选择的区域 {i+1}: 左上角: {pts[0]}, 右下角: {pts[1]}")
    return refPts


def enhance_roi_for_ocr(roi, target_height=60):
    """
    对 ROI 做 OCR 前增强：
      1. 如果文字高度不足 target_height 像素，上采样到目标高度
      2. 转灰度
      3. CLAHE 自适应对比度增强
      4. 锐化滤波
      5. 双边滤波去噪（保边缘）

    返回增强后的灰度图。
    """
    h, w = roi.shape[:2]

    # 1. 上采样：确保文字高度至少 target_height px（Tesseract 推荐 30-60px 字高）
    if h < target_height:
        scale = target_height / h
        new_w = int(w * scale)
        roi = cv2.resize(roi, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        h, w = target_height, new_w

    # 2. 转灰度
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # 3. CLAHE 自适应直方图均衡化（提升对比度，不会像固定阈值那样破坏细节）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. 锐化（增强文字边缘）
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # 5. 双边滤波去噪（保留边缘）
    denoised = cv2.bilateralFilter(sharpened, 5, 50, 50)

    return denoised


def recognize_text_and_save_boxes(img_path, refPts, output_folder,
                                   lang='chi_sim+eng', target_height=60):
    """
    识别图像中的文字（改进版），保存裁剪区域图片并调整大小，然后保存识别结果的框信息。

    参数:
        img_path: 图像文件路径
        refPts: 选择区域的坐标列表
        output_folder: 输出文件夹路径
        lang: Tesseract 语言参数，默认 'chi_sim+eng'
        target_height: 上采样目标文字高度（像素），默认 60px
    """
    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_CMD

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image {img_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for i, (top_left, bottom_right) in enumerate(refPts):
        x1, y1 = top_left
        x2, y2 = bottom_right

        # 裁剪出当前矩形区域（原始图像）
        roi = img[y1:y2, x1:x2]

        # 保存裁剪出的图片（原始，不增强）
        basename = os.path.splitext(os.path.basename(img_path))[0]
        cropped_img_path = os.path.join(output_folder, f'{basename}_crop_{i+1}.png')
        cv2.imwrite(cropped_img_path, roi)
        print(f"Cropped region saved to: {cropped_img_path}")

        # 图像增强（用于 OCR）
        enhanced = enhance_roi_for_ocr(roi, target_height=target_height)

        # 保存增强后的图片用于调试
        enhanced_img_path = os.path.join(output_folder, f'{basename}_enhanced_{i+1}.png')
        cv2.imwrite(enhanced_img_path, enhanced)

        h_enh, w_enh = enhanced.shape[:2]
        h_orig, w_orig = roi.shape[:2]
        scale_x = w_orig / w_enh
        scale_y = h_orig / h_enh

        # ---- 使用 image_to_data 获取每个字符的置信度 ----
        # --oem 1: LSTM only (better accuracy)
        # --psm 6: 假设统一文本块
        config = '--oem 1 --psm 6'

        data = pytesseract.image_to_data(
            enhanced, lang=lang, config=config,
            output_type=pytesseract.Output.DICT
        )

        # 构建 box_info，过滤低置信度字符
        box_info = []
        min_conf = 40  # 最低置信度阈值（0-100）

        for j in range(len(data['text'])):
            text = data['text'][j].strip()
            conf = int(data['conf'][j]) if data['conf'][j] != '-1' else 0

            if not text or conf < min_conf:
                continue

            # 获取边界框（在原增强图上的坐标）
            bx1 = data['left'][j]
            by1 = data['top'][j]
            bx2 = bx1 + data['width'][j]
            by2 = by1 + data['height'][j]

            # 转换坐标到原始 ROI 上的位置
            bx1_orig = int(bx1 * scale_x)
            by1_orig = int(by1 * scale_y)
            bx2_orig = int(bx2 * scale_x)
            by2_orig = int(by2 * scale_y)

            # 再转换到全图坐标
            bx1_full = x1 + bx1_orig
            by1_full = y1 + by1_orig
            bx2_full = x1 + bx2_orig
            by2_full = y1 + by2_orig

            # 过滤明显过大的框（可能是误检）
            char_width = bx2_full - bx1_full
            char_height = by2_full - by1_full
            avg_char_h = (y2 - y1) * 0.8  # 字符高度不应超过 ROI 高度的 80%
            if char_height > avg_char_h and char_height > 100:
                continue

            for ch in text:
                box_info.append({
                    "character": ch,
                    "bbox": [bx1_full, by1_full, bx2_full, by2_full],
                    "conf": conf
                })

        # 保存 Box 信息到 JSON
        box_file_path = os.path.join(output_folder, f'{basename}_box_{i+1}.json')
        with open(box_file_path, 'w', encoding='utf-8') as f:
            json.dump(box_info, f, ensure_ascii=False, indent=2)

        # 打印识别结果
        combined = ''.join(item['character'] for item in box_info)
        print(f"  box{i+1} 识别: {combined}  (共 {len(box_info)} 字符)")


def process_folder(input_folder, refPts, output_folder):
    """
    处理输入文件夹中的所有图片，并根据 refPts 进行文字识别和保存框信息。
    使用原始图像（不二值化），由 enhance_roi_for_ocr 做自适应增强。

    参数:
        input_folder: 原始图片文件夹路径
        refPts: 选择区域的坐标列表
        output_folder: 输出文件夹路径
    """
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)

            # 为每张图片创建单独的输出目录
            image_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(image_output_folder, exist_ok=True)

            # 执行识别并保存结果
            recognize_text_and_save_boxes(img_path, refPts, image_output_folder)
            print(f"Processed {filename} and saved results")


def preprocess_images_in_folder(input_folder, output_folder):
    """
    处理输入文件夹中的所有图片，并将处理后的图片保存到输出文件夹。
    （保留兼容，但OCR主流程不再使用二值化图片）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def preprocess_image(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用自适应阈值代替固定阈值 150（对光照不均更鲁棒）
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            processed_img = preprocess_image(img_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)
            print(f"Processed {filename} and saved to {output_path}")

    print("All images have been processed and saved.")


def process_json_files(folder_path, output_file_path):
    """
    处理文件夹中的所有 JSON 文件，并将每个坐标的信息分别合并，
    打上标签写入一个输出文件。

    改进：过滤掉孤立的低置信度单字符（常见 Tesseract 误检）。
    """
    page_data = {}

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # 提取文件名中的 box 标签
                # 格式: page_X_box_N.json
                box_match = re.search(r'box_(\d+)', file_name)
                box_label = f"box{box_match.group(1)}" if box_match else "box?"

                # 合并字符（保留 conf 用于后处理判断）
                combined_string = ''.join(item['character'] for item in data)

                # 简单后处理：清理明显的误检
                # - 去除行尾孤立的单字符（如多余的 "s", "e"）
                if len(combined_string) > 3:
                    # 检测尾部是否有孤立的单字符（前面是中文，后面是单个英文）
                    # 如 "BERASse" -> "BERAS"（去掉末尾多余字母）
                    # 保守策略：只去掉末尾连续的小写字母（当它们紧跟在数字/大写后）
                    cleaned = re.sub(r'([A-Z\d])[a-z]{1,2}$', r'\1', combined_string)
                    if cleaned != combined_string and len(cleaned) >= len(combined_string) * 0.7:
                        combined_string = cleaned

                labeled_string = f"{box_label}: {combined_string}\n"
                parent_folder = os.path.basename(root)

                if parent_folder in page_data:
                    page_data[parent_folder] += labeled_string
                else:
                    page_data[parent_folder] = labeled_string

    # 将每页的数据写入输出文件
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for parent_folder, combined_string in sorted(page_data.items()):
            output_file.write(f"{parent_folder}:\n{combined_string}\n")

    print(f"结果已保存到: {output_file_path}")
