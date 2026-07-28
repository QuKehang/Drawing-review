import cv2
import json
import os

# 全局变量
refPts = []  # 用于存储所有选择的区域
cropping = False
currentPts = []  # 当前正在选择的区域

# 缩放因子
scale = 0.5

# 鼠标事件回调函数
def click_and_crop(event, x, y, flags, param):
    global cropping, currentPts, refPts

    # 将坐标缩放回原始图像大小
    x = int(x / scale)
    y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        currentPts = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        currentPts.append((x, y))
        cropping = False

        # 在原图上绘制矩形
        cv2.rectangle(image, currentPts[0], currentPts[1], (0, 255, 0), 2)
        # 显示缩放后的图像
        cv2.imshow("image", cv2.resize(image, (0, 0), fx=scale, fy=scale))
        
        # 保存当前选中的区域
        refPts.append(tuple(currentPts))

# 读取图像
image_path = r"C:\Users\ASUS\Mid-term\ProcessingofFixed_Information\images\page_1.png"
image = cv2.imread(image_path)
clone = image.copy()

# 创建窗口并设置大小
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", int(image.shape[1] * scale), int(image.shape[0] * scale))
cv2.setMouseCallback("image", click_and_crop)

# 显示图像并等待按键
print("操作: 鼠标拖动框选区域 | r=重置 | q/ESC=退出")
while True:
    # 显示缩放后的图像
    cv2.imshow("image", cv2.resize(image, (0, 0), fx=scale, fy=scale))
    key = cv2.waitKey(1) & 0xFF

    # 检测窗口是否被手动关闭（点击 X 按钮）
    try:
        if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break  # 窗口已被销毁

    if key == ord("r"):
        # 重置图像
        image = clone.copy()
        refPts.clear()

    elif key == ord("q") or key == 27:  # q 或 ESC 退出
        break

# 打印所有选取的区域
for i, pts in enumerate(refPts):
    print(f"Selected Region {i+1}: Top-left: {pts[0]}, Bottom-right: {pts[1]}")

# 设置保存文件路径
os.makedirs(os.path.dirname(output_path := r"C:\Users\ASUS\Mid-term\ProcessingofFixed_Information\refPts\refPts1.json"), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(refPts, f)

# 关闭所有窗口（Windows 上需要额外的 waitKey 才能真正关闭）
cv2.destroyWindow("image")
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)