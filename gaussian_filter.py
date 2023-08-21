import cv2

# 读取图片
img = cv2.imread('images/tag36h11_005_000.jpg', cv2.IMREAD_COLOR)

# 进行高斯模糊
# 第二个参数是高斯核的尺寸，选择合适的值以达到想要的模糊效果，但需要注意的是这个值必须是奇数，如(15, 15)
# 第三个参数是高斯核的标准差，如果设为0，则函数会自动计算
blurred1 = cv2.GaussianBlur(img, (5, 5), 0)
blurred2 = cv2.GaussianBlur(img, (15, 15), 0)
# # 调整图片尺寸
# resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#
# # 保存图片
# cv2.imwrite('dataset/tag36h11_005_img/tag36h11_005_000.jpg', resized)
cv2.imshow('blurred image1', blurred1)
cv2.imshow('blurred image2', blurred2)
cv2.waitKey(0)
cv2.destroyAllWindows()
