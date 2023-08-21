import cv2

# img = cv2.imread("images/apriltag_origin/tag36_11_00005.jpg")
# img2 = cv2.imread("images/test_4tags2-Camera 74.png")
# print(img2.shape)

import cv2
import numpy as np

# 读取图片
img = cv2.imread('images/apriltag_origin/tag36_11_00005.jpg')

# 获取图片的尺寸
h, w, _ = img.shape

# 计算缩放比例
scale = min(1080 / h, 1920 / w)

# 使用cv2.resize进行缩放
new_w = int(w * scale)
new_h = int(h * scale)
resized_img = cv2.resize(img, (new_w, new_h))

# 创建一个白色的背景
background = np.ones((1080, 1920, 3), np.uint8) * 255

# 获取放置缩放后图像的起始坐标（即背景中心）
start_x = (1920 - new_w) // 2
start_y = (1080 - new_h) // 2

# 将缩放后的图像放在白色背景的中心
background[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img

# 保存或显示图像
cv2.imwrite('images/apriltag_rightsize/tag36_11_00005.jpg', background)
# cv2.imshow('result', background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
