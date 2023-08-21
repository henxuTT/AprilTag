import cv2
import numpy as np

# 读取图像
image = cv2.imread('images/tag36h11_005_000.jpg', cv2.IMREAD_GRAYSCALE)

# 使用 Canny 边缘检测
edges = cv2.Canny(image, 100, 200) # 这里的 100 和 200 是阈值，你可以根据需要调整
print(edges.shape)

# 显示原图和边缘检测后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 保存结果
cv2.imwrite('images/label_edges.jpg', edges)
