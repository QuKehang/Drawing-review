"""
辅助脚本：uv sync 后自动卸载 opencv-python-headless。
该包是 albumentations → albucore 的传递依赖，与 opencv-python 的 GUI 功能冲突。

用法：uv run python scripts/fix_opencv.py
"""
import subprocess
import os


def fix_opencv():
    # 设置 UTF-8 编码，避免 Windows GBK 解码错误
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        import cv2
        # 尝试调用 GUI 函数，检测是否为 headless 版本
        try:
            cv2.namedWindow("_test_", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test_")
            print("[OK] opencv-python GUI 功能正常，无需修复")
            return
        except cv2.error:
            print("[WARN] 检测到 headless 版本，正在卸载...")
    except ImportError:
        print("[SKIP] cv2 未安装")
        return

    result = subprocess.run(
        ["uv", "pip", "uninstall", "opencv-python-headless"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        print("[OK] 已卸载 opencv-python-headless，GUI 功能恢复")
    else:
        err = (result.stderr or result.stdout or "").strip()
        print(f"[FAIL] 卸载失败: {err}")


if __name__ == "__main__":
    fix_opencv()
