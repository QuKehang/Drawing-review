"""
launcher_fixed.py — 桥梁工程设计工具集 统一调度系统
==============================================
一键启动所有子工具，各工具以独立进程运行，互不干扰。

用法：
    python launcher_fixed.py

工具列表：
  ① YOLO 标注裁剪       — 目标检测 + 按标签裁剪
  ② 设计说明判定         — OCR + BERT 分类
  ③ PP-OCR 表格识别      — PaddleOCR 表格结构化识别
  ④ 固定信息提取与对比   — 区域选取 + OCR + Excel 比对（修正版）
  ⑤ 保护层厚度测量       — U-Net 分割 + 交互式测量

与 launcher.py 的区别：
  - 工具④ 指向 main_gui_v2_fixed.py（含对比逻辑修正 + OCR前自动清理缓存）
"""

import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QGroupBox,
    QMessageBox, QComboBox
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QProcess

# ============================================================
#  工具注册表
# ============================================================
MIDTERM_DIR = os.path.dirname(os.path.abspath(__file__))

TOOLS = [
    {
        "id": "crop_by_label",
        "name": "YOLO 标注裁剪",
        "desc": "使用 YOLOv5 ONNX 模型检测图纸中的\n图名、标注、标题栏等区域并自动裁剪分类",
        "icon": "🎯",
        "color": "#0078d4",
        "script": os.path.join(MIDTERM_DIR, "Location", "Drawing_location.py"),
        "cwd": os.path.join(MIDTERM_DIR, "Location"),
    },
    {
        "id": "judge_crop",
        "name": "设计说明判定",
        "desc": "对裁剪后的标注文本进行 OCR 识别，\n使用 BERT 模型判定是否符合设计说明规范",
        "icon": "📋",
        "color": "#107c10",
        "script": os.path.join(MIDTERM_DIR, "Annotation-test", "Annotation_check_without.py"),
        "cwd": os.path.join(MIDTERM_DIR, "Annotation-test"),
    },
    {
        "id": "pp_ocr",
        "name": "PP-OCR 表格识别",
        "desc": "基于 PaddleOCR 的表格\n结构化识别，支持 JSON 标注/直接识别模式",
        "icon": "📊",
        "color": "#d83b01",
        "script": os.path.join(MIDTERM_DIR, "PP-OCR _table_reading", "Table_Recognition.py"),
        "cwd": os.path.join(MIDTERM_DIR, "PP-OCR _table_reading"),
    },
    {
        "id": "fixed_info",
        "name": "固定信息提取与对比",
        "desc": "OpenCV 交互式区域选取 → Tesseract OCR\n→ 与 Excel 参考表自动比对图名图号",
        "icon": "🔍",
        "color": "#6b3fa0",
        "script": os.path.join(MIDTERM_DIR, "ProcessingofFixed_Information", "Completeness_check.py"),
        "cwd": os.path.join(MIDTERM_DIR, "ProcessingofFixed_Information"),
    },
]


# ============================================================
#  工具卡片组件
# ============================================================
class ToolCard(QFrame):
    """单个工具的卡片：图标 + 标题 + 描述 + 启动按钮"""

    def __init__(self, tool_info, launcher=None, parent=None):
        super().__init__(parent)
        self.tool_info = tool_info
        self.launcher = launcher
        self.process = None
        self._build_ui()

    def _build_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet(f"""
            ToolCard {{
                background: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 15px;
            }}
            ToolCard:hover {{
                border-color: {self.tool_info['color']};
            }}
        """)
        self.setMinimumSize(280, 220)
        self.setMaximumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 图标 + 标题
        header = QHBoxLayout()
        icon_label = QLabel(self.tool_info["icon"])
        icon_label.setFont(QFont("Segoe UI Emoji", 28))
        icon_label.setFixedWidth(50)
        header.addWidget(icon_label)

        name_label = QLabel(self.tool_info["name"])
        name_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        name_label.setStyleSheet(f"color: {self.tool_info['color']}; border: none; background: transparent;")
        header.addWidget(name_label)
        header.addStretch()
        layout.addLayout(header)

        # 描述
        desc_label = QLabel(self.tool_info["desc"])
        desc_label.setFont(QFont("Microsoft YaHei", 9))
        desc_label.setStyleSheet("color: #666; border: none; background: transparent; padding-left: 50px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        layout.addStretch()

        # 启动按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.launch_btn = QPushButton("▶  启 动")
        self.launch_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.launch_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.tool_info['color']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 28px;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
            QPushButton:pressed {{
                background-color: #555;
            }}
        """)
        self.launch_btn.clicked.connect(self._launch_tool)
        btn_layout.addWidget(self.launch_btn)
        layout.addLayout(btn_layout)

    def _launch_tool(self):
        """用独立进程启动子工具（关闭窗口后按钮自动恢复）"""
        script = self.tool_info["script"]
        cwd = self.tool_info["cwd"]

        if not os.path.isfile(script):
            QMessageBox.critical(
                self, "错误",
                f"找不到脚本文件:\n{script}\n\n请确认文件路径正确。"
            )
            return

        # 使用启动器选择的 Python 环境
        python_exe = self.launcher.get_python_exe() if self.launcher else sys.executable

        try:
            # 使用 QProcess 替代 subprocess，可监听 finished 信号
            self.process = QProcess(self)
            self.process.setWorkingDirectory(cwd)
            self.process.finished.connect(self._on_tool_closed)
            self.process.start(python_exe, [script])

            self.launch_btn.setText("●  运行中")
            self.launch_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #888;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 28px;
                }}
            """)
            self.launch_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(
                self, "启动失败",
                f"无法启动 {self.tool_info['name']}:\n\n{e}"
            )

    def _on_tool_closed(self, exit_code, exit_status):
        """子工具窗口关闭后自动恢复按钮"""
        self.reset_button()

    def reset_button(self):
        """重置按钮状态"""
        self.launch_btn.setText("▶  启 动")
        self.launch_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.tool_info['color']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 28px;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        self.launch_btn.setEnabled(True)


# ============================================================
#  主窗口
# ============================================================
class LauncherMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("桥梁工程图纸智能审查")
        self.setMinimumSize(1000, 700)
        self.cards: list[ToolCard] = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(30, 20, 30, 20)

        # ---- 标题区 ----
        title = QLabel("🛠  桥梁工程图纸智能审查")
        title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin-bottom: 5px;")
        main_layout.addWidget(title)

        subtitle = QLabel("统一调度平台 — 选择一个工具开始工作")
        subtitle.setFont(QFont("Microsoft YaHei", 10))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888; margin-bottom: 15px;")
        main_layout.addWidget(subtitle)

        # ---- 分隔线 ----
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #e0e0e0; max-height: 1px;")
        main_layout.addWidget(sep)

        # ---- 环境选择 ----
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("Python 环境:"))
        self.env_combo = QComboBox()
        self.env_combo.setMinimumWidth(300)
        self.env_combo.addItem(f"当前环境 ({sys.executable})", sys.executable)
        # 尝试添加常见的 conda 环境
        for env_name, env_pattern in [
            ("env03 (PaddleOCR 3.x)", "env03"),
            ("base", "base"),
        ]:
            conda_dir = os.path.dirname(os.path.dirname(sys.executable))
            candidate = os.path.join(conda_dir, env_pattern, "python.exe")
            if os.path.isfile(candidate) and candidate != sys.executable:
                self.env_combo.addItem(f"{env_name} ({candidate})", candidate)
        self.env_combo.currentIndexChanged.connect(self._on_env_changed)
        env_layout.addWidget(self.env_combo)
        env_layout.addStretch()

        env_group = QGroupBox("运行配置")
        env_group.setLayout(env_layout)
        env_group.setStyleSheet("QGroupBox { font-size: 11px; color: #666; }")
        main_layout.addWidget(env_group)

        # ---- 工具卡片网格 ----
        grid = QGridLayout()
        grid.setSpacing(20)

        for i, tool in enumerate(TOOLS):
            card = ToolCard(tool, launcher=self)
            self.cards.append(card)
            row, col = divmod(i, 3)
            grid.addWidget(card, row, col)

        main_layout.addLayout(grid)
        main_layout.addStretch()

        # ---- 底部状态栏 ----
        footer = QLabel("各工具以独立进程运行，关闭本窗口不影响已启动的工具。")
        footer.setFont(QFont("Microsoft YaHei", 9))
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #aaa; margin-top: 10px;")
        main_layout.addWidget(footer)

    def get_python_exe(self):
        """返回当前选择的 Python 可执行文件路径"""
        return self.env_combo.currentData() or sys.executable

    def _on_env_changed(self, index):
        """切换 Python 环境时重置所有卡片按钮"""
        for card in self.cards:
            card.reset_button()


# ============================================================
#  入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-size: 11px;
            color: #555;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 5px;
        }
        QComboBox {
            padding: 4px 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
    """)

    window = LauncherMain()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
