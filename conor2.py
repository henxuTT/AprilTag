import cv2
import numpy as np

# 鼠标事件回调函数
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        for corner in corners:
            x_corner, y_corner = corner.ravel()
            # 计算鼠标位置与角点位置的距离
            distance = np.sqrt((x - x_corner) ** 2 + (y - y_corner) ** 2)
            if distance < 5:  # 当鼠标悬停在角点附近5像素范围内时，显示像素坐标
                print("Corner pixel coordinates: ({}, {})".format(x_corner, y_corner))
                break

# 读取AprilTag图像
# img = cv2.imread('./images/tag36h11_005_000.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图像
# img = cv2.imread('roi.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图像
img = cv2.imread('images/tag36h11_005.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
print(img.shape)
# 设定需要检测的角点数量
max_corners = 40

# 设定角点检测的质量等级和最小距离
quality_level = 0.01
min_distance = 10

# 进行角点检测
corners = cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)

# 将角点坐标转换为整数
corners = np.intp(corners)

with open('tag36h11_005_000.txt', 'w') as file:
    for corner in corners:
        x, y = corner.ravel()
        file.write("{} {}\n".format(x, y))

# 创建窗口并设置鼠标事件回调函数
cv2.namedWindow('AprilTag with Corners')
cv2.setMouseCallback('AprilTag with Corners', show_coordinates)

# 显示AprilTag图像并绘制角点
img_with_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为3通道图像
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img_with_corners, (x, y), 5, (0, 0, 255), -1)

# 显示结果图像
cv2.imshow('AprilTag with Corners', img_with_corners)

# 等待按下任意键关闭窗口S
cv2.waitKey(0)
cv2.destroyAllWindows()
