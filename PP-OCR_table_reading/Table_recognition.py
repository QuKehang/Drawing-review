import os
import sys
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    _torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
    import torch  # noqa: F401 — 必须在 paddleocr 之前导入

import cv2
import json
import threading
import io
import traceback
import numpy as np
import re
from copy import deepcopy

# =====================================================================
# 中文路径兼容（OpenCV C++ 后端不支持 Unicode 路径）
# =====================================================================
def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """读取图像，支持中文路径"""
    data = np.fromfile(path, dtype=np.uint8)
    if data is None:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str, img) -> bool:
    """保存图像，支持中文路径"""
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    buf.tofile(path)
    return True


# =====================================================================
# 环境检查 (在 GUI 启动前完成)
# =====================================================================
PADDLE_AVAILABLE = False
IMPORT_ERROR_MSG = ""

try:
    from paddleocr import PPStructure
    PADDLE_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR_MSG = (
        f"导入 paddleocr 失败！\n\n"
        f"错误: {e}\n\n"
        f"请确认:\n"
        f"  1. paddleocr 已安装: pip install paddleocr==2.10.0\n"
        f"  2. paddlepaddle 已安装: pip install paddlepaddle==2.6.2\n"
        f"  3. 当前使用 env01 环境 (PaddleOCR 2.x)\n\n"
        f"  4. 安装必要依赖: pip install premailer opencv-python"
    )

# =====================================================================
# 路径配置
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_IMG_FOLDER = os.path.join(SCRIPT_DIR, "Table_Images")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "Output")

# =====================================================================
# tkinter 导入
# ====================================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# =====================================================================
# 输出重定向 —— 将 print 输出同时显示到 GUI 文本框
# =====================================================================
class TextRedirector(io.StringIO):
    """将 stdout 重定向到 tkinter Text 控件"""

    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s: str):
        # 写入原始 stdout
        sys.__stdout__.write(s)
        # 写入 GUI 文本框
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        sys.__stdout__.flush()


# =====================================================================
# 图像预处理
# =====================================================================
def preprocess_image(img, intensity="medium"):
    """
    对图像进行自适应预处理，提升 OCR 识别准确率。

    处理流程：
      1. 转换到 LAB 色彩空间，对 L 通道做 CLAHE 增强对比度
      2. Unsharp mask 锐化使文字边缘更清晰

    参数：
        img: BGR 格式的 numpy 数组
        intensity: "mild" / "medium"

    返回：
        预处理后的 BGR 图像
    """
    cfg = {
        "mild":   {"clahe_clip": 1.5, "clahe_grid": (12, 12), "sharpen": False},
        "medium": {"clahe_clip": 2.5, "clahe_grid": (8, 8),   "sharpen": True},
    }
    c = cfg.get(intensity, cfg["medium"])

    # 1. CLAHE 对比度增强（LAB L 通道）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=c["clahe_clip"], tileGridSize=c["clahe_grid"])
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 2. Unsharp mask 锐化
    if c["sharpen"]:
        blur = cv2.GaussianBlur(result, (0, 0), sigmaX=2.0)
        result = cv2.addWeighted(result, 1.8, blur, -0.8, 0)

    return result

SPEED_PRESETS = {
    # ---- 快速：降低分辨率 + 跳过后处理（精度有损） ----
    "fast": {
        "text_det_limit_side_len": 480,
        "text_det_thresh": 0.3,
        "text_det_box_thresh": 0.5,
        "text_det_unclip_ratio": 1.5,
        "text_rec_score_thresh": 0.3,
        "use_textline_orientation": False,
        "use_preprocessing": False,
        "use_fallback": False,
    },
    # ---- 均衡：中等分辨率 + 保留预处理（精度轻微损失） ----
    "balanced": {
        "text_det_limit_side_len": 640,
        "text_det_thresh": 0.25,
        "text_det_box_thresh": 0.4,
        "text_det_unclip_ratio": 1.2,
        "text_rec_score_thresh": 0.1,
        "use_textline_orientation": True,
        "use_preprocessing": True,
        "use_fallback": True,
    },
    # ---- 准确：完整精度，与原始 PP-OCRv4 参数一致 ----
    "accurate": {
        "text_det_limit_side_len": 960,
        "text_det_thresh": 0.25,
        "text_det_box_thresh": 0.4,
        "text_det_unclip_ratio": 1.2,
        "text_rec_score_thresh": 0.1,
        "use_textline_orientation": True,
        "use_preprocessing": True,
        "use_fallback": True,
    },
}

# PPStructure 2.x 构造函数参数（取代 3.x 的 .predict() 运行时参数）
# 速度预设 → 引擎构造参数映射
SPEED_ENGINE_PARAMS = {
    "fast": {
        "det_limit_side_len": 480,
        "det_limit_type": "min",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.5,
        "det_db_unclip_ratio": 1.5,
    },
    "balanced": {
        "det_limit_side_len": 640,
        "det_limit_type": "min",
        "det_db_thresh": 0.25,
        "det_db_box_thresh": 0.4,
        "det_db_unclip_ratio": 1.2,
    },
    "accurate": {
        "det_limit_side_len": 960,
        "det_limit_type": "min",
        "det_db_thresh": 0.25,
        "det_db_box_thresh": 0.4,
        "det_db_unclip_ratio": 1.2,
    },
}


def extract_confidence_scores(result):
    """从 PPStructure 结果中提取置信度列表（dict 格式，每项有 score 字段）"""
    if not result:
        return []
    try:
        return [item.get("score", 0.0) for item in result if isinstance(item, dict)]
    except Exception:
        return []


def compute_mean_confidence(result):
    """计算识别结果的平均置信度，无数据时返回 1.0"""
    scores = extract_confidence_scores(result)
    if not scores:
        return 1.0
    return float(np.mean(scores))


# =====================================================================
# 两阶段自适应识别
# =====================================================================
def recognize_with_fallback(img, engine, preprocessing_enabled=True, log_func=print,
                           speed="balanced"):
    """
    两阶段自适应识别：
      Pass 1：轻度预处理（或跳过），调用 engine(img)
      若平均置信度 < 0.7 且启用回退 → Pass 2：medium 预处理后再次识别
      返回较优结果

    参数：
        speed: "fast" / "balanced" / "accurate"
    """
    preset = SPEED_PRESETS.get(speed, SPEED_PRESETS["balanced"])
    use_fallback = preset["use_fallback"]

    metadata = {
        "confidence_before": None,
        "confidence_after": None,
        "preprocessing_used": "none",
        "fallback_triggered": False,
    }

    # ---- Pass 1 ----
    if preprocessing_enabled and preset["use_preprocessing"]:
        img_pass1 = preprocess_image(img, intensity="mild")
        metadata["preprocessing_used"] = "mild"
    else:
        img_pass1 = img

    result = engine(img_pass1, return_ocr_result_in_table=True)
    conf1 = compute_mean_confidence(result)
    metadata["confidence_before"] = conf1
    log_func(f"  Pass 1 平均置信度: {conf1:.4f}")

    # ---- Pass 2：低置信度时回退 ----
    if use_fallback and conf1 < 0.7 and metadata["preprocessing_used"] == "mild":
        log_func(f"  置信度 {conf1:.4f} < 阈值 0.7，使用 medium 预处理重试...")
        metadata["fallback_triggered"] = True

        img_pass2 = preprocess_image(img, intensity="medium")
        result2 = engine(img_pass2, return_ocr_result_in_table=True)
        conf2 = compute_mean_confidence(result2)
        metadata["confidence_after"] = conf2
        metadata["preprocessing_used"] = "medium"
        log_func(f"  Pass 2 平均置信度: {conf2:.4f}")

        if conf2 > conf1:
            log_func(f"  采用 Pass 2 结果 (提升 {conf2 - conf1:.4f})")
            return result2, metadata
        else:
            log_func(f"  保持 Pass 1 结果 (Pass 2 未改善)")
            metadata["preprocessing_used"] = "mild"
            return result, metadata

    return result, metadata


# =====================================================================
# 后处理：图号破折号补全
# =====================================================================
KNOWN_AREA_CODES = ['00', '01', '02', '10', '11', '12', '14', '20']


def postprocess_figure_number(text):
    """
    修复图号列常见的 OCR 错误，例如：
      "12401 ~ 404"   →  "12-401 ~ 404"
      "12101 ~ 104"   →  "12-101 ~ 104"
      "14001 ~ 013"   →  "14-001 ~ 013"
      "12201 ~ 209"   →  "12-201 ~ 209"
    """
    for code in KNOWN_AREA_CODES:
        # 模式 1：XXNNN ~ NNN（空格波浪号空格，缺破折号）
        p1 = re.compile(r'\b(' + code + r')(\d{3})\s*~\s*(\d{3})\b')
        if p1.search(text):
            text = p1.sub(r'\1-\2 ~ \3', text)
            continue

        # 模式 2：XXNNN-NNN（有横杠分隔但区号后缺破折号）
        p2 = re.compile(r'\b(' + code + r')(\d{3})-(\d{3})\b')
        if p2.search(text):
            text = p2.sub(r'\1-\2-\3', text)
            continue

        # 模式 3：XXNNN 独立出现（无波浪号、无横杠），且不在已经修正的范围内
        # 在已含波浪号或已含横杠的文本中跳过
        if '~' not in text and '-' not in text:
            p3 = re.compile(r'\b(' + code + r')(\d{3})\b')
            text = p3.sub(r'\1-\2', text)

    return text


def postprocess_table_html(html_content, log_func=print):
    """
    对 PPStructure 输出的 HTML 表格内容进行后处理。
    目前对 <td> 中的疑似图号文本应用破折号补全。
    """
    def fix_td(match):
        inner = match.group(1)
        # 仅处理以 2 位数字开头的疑似图号
        if re.match(r'^\d{2}\d{3,}', inner.strip()):
            fixed = postprocess_figure_number(inner.strip())
            if fixed != inner.strip():
                log_func(f"  图号修正: '{inner.strip()}' -> '{fixed}'")
            return f'<td>{fixed}</td>'
        return match.group(0)

    return re.sub(r'<td>(.*?)</td>', fix_td, html_content)


# =====================================================================
# 表格 HTML 重建 —— 修复 PP-Structure 模型的列合并问题
# =====================================================================
def rebuild_table_html(boxes, rec_res):
    
    if not boxes or not rec_res:
        return "<html><body><table></table></body></html>"

    n = len(boxes)

    # ---- 1. 行检测：按 y 中心聚类，间隔 > 15px 为新行 ----
    y_centers = [(b[1] + b[3]) / 2 for b in boxes]
    y_sorted = sorted(set(round(y) for y in y_centers))

    row_breaks = [y_sorted[0]]
    prev_y = y_sorted[0]
    for y in y_sorted[1:]:
        if y - prev_y > 15:
            row_breaks.append(y)
        prev_y = y
    n_rows = len(row_breaks)

    # ---- 2. 列检测：按 x 中心聚类，间隔 ≥ 35px 为新列 ----
    x_centers = sorted(set(round((b[0] + b[2]) / 2) for b in boxes))
    col_groups = []
    current = [x_centers[0]]
    for x in x_centers[1:]:
        if x - current[-1] < 35:
            current.append(x)
        else:
            col_groups.append(current)
            current = [x]
    col_groups.append(current)
    col_centers = [sum(g) / len(g) for g in col_groups]
    n_cols = len(col_groups)

    # ---- 3. 将每个 cell 分配到 (row, col) 网格 ----
    def _find_col(cx):
        for ci, cc in enumerate(col_centers):
            if ci < n_cols - 1:
                if cx < (cc + col_centers[ci + 1]) / 2:
                    return ci
            else:
                return ci
        return n_cols - 1

    def _find_row(cy):
        for ri in range(n_rows):
            if ri < n_rows - 1:
                if cy < (row_breaks[ri] + row_breaks[ri + 1]) / 2:
                    return ri
            else:
                return ri
        return n_rows - 1

    grid = {}
    for i in range(n):
        cy = (boxes[i][1] + boxes[i][3]) / 2
        cx = (boxes[i][0] + boxes[i][2]) / 2
        ri = _find_row(cy)
        ci = _find_col(cx)
        text = rec_res[i][0] if i < len(rec_res) else ""
        grid[(ri, ci)] = text

    # ---- 4. 构建 HTML ----
    parts = ["<html><body><table>"]
    # 表头（第 0 行）
    parts.append("<thead><tr>")
    for ci in range(n_cols):
        parts.append(f"<td>{grid.get((0, ci), '')}</td>")
    parts.append("</tr></thead>")
    # 表体（其余行）
    parts.append("<tbody>")
    for ri in range(1, n_rows):
        parts.append("<tr>")
        for ci in range(n_cols):
            parts.append(f"<td>{grid.get((ri, ci), '')}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></body></html>")
    return "".join(parts)


def fix_table_results(result):
    """对 PPStructure 识别结果中的每个 table 区域，用 bbox 数据重建 HTML。

    原地修改 result 列表中的每个 table dict。
    """
    if not result:
        return result
    for item in result:
        if item.get("type") != "table":
            continue
        res = item.get("res")
        if not isinstance(res, dict):
            continue
        boxes = res.get("boxes", [])
        rec_res = res.get("rec_res", [])
        if boxes and rec_res:
            res["html"] = rebuild_table_html(boxes, rec_res)
    return result


# =====================================================================
# 核心处理逻辑
# =====================================================================
def init_table_engine(log_func=print, speed="accurate"):
    """初始化 PPStructure 表格识别引擎 (PP-Structure v1 + PP-OCRv4 中文模型)

    使用 PP-Structure v1 表格模型 (PaddlePaddle 2.x 兼容) + PP-OCRv4 中文 OCR。
    SLANet (PP-StructureV2) 模型使用 PaddlePaddle 3.x 格式，Paddle 2.6.2 无法加载。
    """
    log_func("正在初始化 PPStructure 表格引擎 (PP-Structure v1 + PP-OCRv4)...")

    # ---- 获取缓存的 PP-OCRv4 中文模型路径 ----
    # PP-Structure v1 仅有英文表格模型，需要手动指定中文 OCR 模型和字典
    _paddleocr_home = os.path.join(os.path.expanduser("~"), ".paddleocr", "whl")
    _ch_det_dir = os.path.join(_paddleocr_home, "det", "ch", "ch_PP-OCRv4_det_infer")
    _ch_rec_dir = os.path.join(_paddleocr_home, "rec", "ch", "ch_PP-OCRv4_rec_infer")
    # 中文识别字典（覆盖 lang="en" 的英文字典，确保中文字符可识别）
    _pocr_pkg = sys.modules.get("paddleocr")
    if _pocr_pkg is not None:
        _ch_dict_path = os.path.join(
            os.path.dirname(_pocr_pkg.__file__), "ppocr", "utils", "ppocr_keys_v1.txt"
        )
    else:
        _ch_dict_path = None

    # ---- 速度相关的检测参数 ----
    speed_params = SPEED_ENGINE_PARAMS.get(speed, SPEED_ENGINE_PARAMS["accurate"])

    engine = PPStructure(
        structure_version="PP-Structure",     # v1 (Paddle 2.x 兼容，非 SLANet)
        lang="en",                             # v1 仅有英文表格模型
        ocr_version="PP-OCRv4",
        # 手动指定中文 OCR 模型和字典，覆盖 lang="en" 的英文 OCR
        det_model_dir=_ch_det_dir if os.path.isdir(_ch_det_dir) else None,
        rec_model_dir=_ch_rec_dir if os.path.isdir(_ch_rec_dir) else None,
        rec_char_dict_path=_ch_dict_path if os.path.isfile(_ch_dict_path) else None,
        # ---- 管线开关 ----
        table=True,
        layout=True,                           # 需要 layout 以启用 OCR
        ocr=True,
        formula=False,
        show_log=False,
        # ---- 速度参数 ----
        **speed_params,
    )
    log_func(f"引擎初始化完成（PP-Structure v1 表格 + PP-OCRv4 中文 OCR, speed={speed}）。")
    return engine


def _save_table_results(res, save_folder, img_name, log_func=print):
    """保存表格识别结果为 Excel + HTML + JSON 摘要"""
    import pandas as pd

    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)

    # 统计表格数量，用于编号
    table_idx = 0

    # 保存每个区域的 JSON 信息
    res_lines = []
    for region in deepcopy(res):
        roi_img = region.pop("img", None)
        res_lines.append(json.dumps(region, ensure_ascii=False))

        if (region["type"].lower() == "table"
                and len(region.get("res", {})) > 0
                and "html" in region["res"]):
            table_idx += 1
            # ---- 后处理：图号修正 ----
            html = region["res"]["html"]
            html = postprocess_table_html(html, log_func=log_func)

            # ---- 保存 Excel：{img_name}_{序号}.xlsx ----
            excel_path = os.path.join(
                excel_save_folder, f"{img_name}_{table_idx}.xlsx"
            )
            try:
                # 优先使用 tablepyxl（保留样式）
                try:
                    from tablepyxl import tablepyxl
                    tablepyxl.document_to_xl(html, excel_path)
                except ImportError:
                    # 回退到 pandas
                    dfs = pd.read_html(html)
                    if dfs:
                        dfs[0].to_excel(excel_path, index=False)
                    else:
                        log_func(f"  警告: 无法从 HTML 解析表格数据")
                        continue
                log_func(f"  -> 表格 {table_idx} 已保存: {excel_path}")
            except Exception as e:
                log_func(f"  错误: 保存表格 {table_idx} 失败: {e}")

        elif region["type"].lower() == "figure" and roi_img is not None:
            fig_path = os.path.join(
                excel_save_folder, f"{img_name}_fig_{table_idx}.jpg"
            )
            imwrite_unicode(fig_path, roi_img)

    # ---- 保存识别文本摘要 ----
    txt_path = os.path.join(excel_save_folder, f"{img_name}_res.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(res_lines) + "\n")
        log_func(f"  -> 识别摘要已保存: {txt_path}")
    except Exception as e:
        log_func(f"  错误: 保存识别摘要失败: {e}")

    return excel_save_folder


def run_direct_mode(img_folder, results_dir, preprocessing_enabled=True, log_func=print,
                    speed="balanced", progress_callback=None):
    """直接识别模式（全图表格识别）"""
    log_func("\n=== 表格识别 ===")
    log_func(f"速度预设: {speed} | 预处理: {'开' if preprocessing_enabled else '关'}")

    if not os.path.exists(img_folder):
        log_func(f"错误: 图像文件夹不存在 -> {img_folder}")
        return

    img_files = [f for f in os.listdir(img_folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not img_files:
        log_func(f"在 {img_folder} 中没有找到图像文件。")
        return

    total = len(img_files)
    log_func(f"找到 {total} 张图片")

    # 初始化引擎（传入速度预设）
    engine = init_table_engine(log_func, speed=speed)

    success_count = 0
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(img_folder, img_file)
        log_func(f"\n处理 ({i + 1}/{total}): {img_file}")
        img = imread_unicode(img_path)
        if img is None:
            log_func(f"  无法读取图像: {img_file}")
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        try:
            result, metadata = recognize_with_fallback(
                img, engine,
                preprocessing_enabled=preprocessing_enabled,
                log_func=log_func,
                speed=speed,
            )

            img_name = os.path.splitext(img_file)[0]
            if result and len(result) > 0:
                fix_table_results(result)  # 修复列合并问题
                save_path = _save_table_results(result, results_dir, img_name, log_func=log_func)
                log_func(f"  -> 结果已保存: {save_path} ({len(result)} 个区域)")
                success_count += 1
            else:
                log_func(f"  -> 未检测到表格结构")
        except Exception as e:
            log_func(f"  识别失败: {e}")

        if progress_callback:
            progress_callback(i + 1, total)

    log_func(f"\n处理完成: {success_count}/{total} 张图片成功识别")

# =====================================================================
# GUI 应用程序类
# =====================================================================
class PPOCRApp:
    """PP-OCR 表格识别 GUI 应用 (PP-OCRv4)"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PP-OCR 表格识别工具 (v4 — PP-OCRv4)")
        self.root.geometry("780x680")
        self.root.minsize(700, 600)

        # 处理中标志
        self.processing = False

        # 初始化引擎引用
        self.engine = None

        self._build_ui()

        # 处理窗口关闭
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """构建 GUI 界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== 标题 =====
        title_label = ttk.Label(
            main_frame,
            text="图纸表格识别模块",
            font=("Microsoft YaHei", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # ===== 路径配置 =====
        path_frame = ttk.LabelFrame(main_frame, text="路径设置", padding="10")
        path_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        # 图片文件夹
        ttk.Label(path_frame, text="图片文件夹:").grid(row=0, column=0, sticky="w", pady=5)
        self.img_var = tk.StringVar(value=DEFAULT_IMG_FOLDER)
        ttk.Entry(path_frame, textvariable=self.img_var, width=55).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(path_frame, text="浏览...",
                   command=lambda: self._browse_folder(self.img_var)).grid(row=0, column=2, pady=5)

        # 结果保存路径
        ttk.Label(path_frame, text="结果保存路径:").grid(row=1, column=0, sticky="w", pady=5)
        self.results_var = tk.StringVar(value=DEFAULT_RESULTS_DIR)
        ttk.Entry(path_frame, textvariable=self.results_var, width=55).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(path_frame, text="浏览...",
                   command=lambda: self._browse_folder(self.results_var)).grid(row=1, column=2, pady=5)

        # ===== 速度预设 =====
        speed_frame = ttk.LabelFrame(main_frame, text="速度预设", padding="10")
        speed_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        self.speed_var = tk.StringVar(value="accurate")
        speed_desc = {
            "accurate": "准确 (推荐) — 完整精度，高分辨率(960) + 预处理 + 回退 + 方向校正",
            "balanced": "均衡 — 精度轻微损失，中分辨率(640) + 预处理 + 回退      (~2x 加速)",
            "fast": "快速 — 精度有损，低分辨率(480) + 无预处理 + 无回退         (~4x 加速)",
        }
        row_offset = 0
        for key, desc in speed_desc.items():
            ttk.Radiobutton(
                speed_frame, text=desc,
                variable=self.speed_var, value=key,
            ).grid(row=row_offset, column=0, sticky="w", pady=1)
            row_offset += 1

        # ===== 预处理控制 =====
        preproc_frame = ttk.LabelFrame(main_frame, text="图像预处理", padding="10")
        preproc_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        self.preproc_var = tk.BooleanVar(value=True)
        self.preproc_check = ttk.Checkbutton(
            preproc_frame,
            text="启用图像预处理 (CLAHE 对比度增强 + 锐化，快速模式下自动禁用)",
            variable=self.preproc_var,
        )
        self.preproc_check.grid(row=0, column=0, sticky="w")

        # ===== 操作按钮 =====
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=(5, 10))

        self.start_btn = ttk.Button(
            btn_frame, text="▶  开始处理",
            command=self._start_processing,
            width=20
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="■  停止",
            command=self._stop_processing,
            state=tk.DISABLED,
            width=10
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 进度条（按图片处理比例显示）
        self.progress = ttk.Progressbar(btn_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.LEFT, padx=20)

        # ===== 日志输出区 =====
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="5")
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(5, 0))

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=85,
            height=18,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 配置行权重 (日志区可扩展)
        main_frame.rowconfigure(5, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 初始日志
        if PADDLE_AVAILABLE:
            self._log("PP-OCR 表格识别工具已就绪 (PaddleOCR 2.10.0 + PP-OCRv4 中文 OCR)。\n")
            self._log("使用 PP-Structure v1 表格模型 (PaddlePaddle 2.x 兼容)。\n")
            self._log("默认使用「准确」模式，可在「速度预设」中切换更快档位。\n")
            self._log("请选择识别模式并确认路径，然后点击 [开始处理] 按钮。\n")
        else:
            self._log("=" * 55 + "\n")
            self._log(IMPORT_ERROR_MSG + "\n")
            self._log("=" * 55 + "\n")
            self.start_btn.config(state=tk.DISABLED)

    def _browse_folder(self, var: tk.StringVar):
        """打开文件夹选择对话框"""
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _log(self, text: str):
        """向日志区追加文本"""
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()

    def _start_processing(self):
        """开始处理 (在后台线程中运行)"""
        if self.processing:
            return

        # 验证路径
        img_folder = self.img_var.get().strip()

        if not img_folder:
            messagebox.showwarning("路径错误", "请选择图片文件夹。")
            return

        # 确认
        if not messagebox.askokcancel("确认", "是否开始处理？\n\n此操作可能需要数分钟，取决于图片大小和数量。"):
            return

        # 更新 UI 状态
        self.processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress["value"] = 0
        self.progress["maximum"] = 100
        self.log_text.delete(1.0, tk.END)
        self._log("=" * 55 + "\n")
        self._log("开始处理...\n")
        self._log("=" * 55 + "\n")

        # 在后台线程中运行
        thread = threading.Thread(target=self._run_processing, daemon=True)
        thread.start()

    def _stop_processing(self):
        """停止处理"""
        self.processing = False
        self._log("\n⚠ 用户停止了处理。\n")
        self._reset_ui()

    def _set_progress(self, current: int, total: int):
        """从后台线程安全更新进度条"""
        def _update():
            self.progress["maximum"] = total
            self.progress["value"] = current
        self.root.after(0, _update)

    def _run_processing(self):
        """后台处理线程"""
        try:
            img_folder = self.img_var.get().strip()
            results_dir = self.results_var.get().strip()

            # 读取设置
            speed = self.speed_var.get()
            preprocessing_enabled = self.preproc_var.get()

            preset = SPEED_PRESETS.get(speed, SPEED_PRESETS["balanced"])
            self._log(f"速度预设: {speed} | limit_side_len={SPEED_ENGINE_PARAMS[speed]['det_limit_side_len']} | "
                      f"预处理={preset['use_preprocessing']} | 回退={preset['use_fallback']}\n")

            # 确保目录存在
            os.makedirs(results_dir, exist_ok=True)

            def progress_callback(current, total):
                self._set_progress(current, total)

            run_direct_mode(img_folder, results_dir,
                            preprocessing_enabled=preprocessing_enabled,
                            log_func=self._log,
                            speed=speed,
                            progress_callback=progress_callback)

            if self.processing:
                self._log("\n" + "=" * 55 + "\n")
                self._log(f"全部处理完成！结果保存在: {results_dir}\n")
                self._log("=" * 55 + "\n")
                # 弹窗提示
                self.root.after(0, lambda: messagebox.showinfo(
                    "处理完成", f"识别完成！\n\n结果保存在:\n{results_dir}"
                ))
        except Exception as e:
            self._log(f"\n发生未预期错误: {e}\n")
            self._log(traceback.format_exc())
            self.root.after(0, lambda e=e: messagebox.showerror(
                "处理失败", f"发生错误:\n{e}"
            ))
        finally:
            self._reset_ui()

    def _reset_ui(self):
        """恢复 UI 到就绪状态"""
        self.processing = False
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.progress.config(value=0))

    def _on_close(self):
        """关闭窗口"""
        if self.processing:
            if not messagebox.askokcancel("确认退出", "正在处理中，确定要退出吗？"):
                return
            self.processing = False
        self.root.destroy()


# =====================================================================
# 主入口
# =====================================================================
def main():
    root = tk.Tk()

    # 设置样式
    style = ttk.Style()
    try:
        style.theme_use("clam")  # 更现代的观感
    except tk.TclError:
        pass  # 某些平台可能没有 clam 主题

    app = PPOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
