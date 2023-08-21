import cv2
import numpy as np
import os



def test_match(img_path, label_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为3通道图像

    # 读取txt文件并绘制圆点
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x, y = line.split()
            x = int(float(x))  # 从字符串转换为整数
            y = int(float(y))  # 从字符串转换为整数
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # 在对应的坐标上绘制小圆点

    # 显示结果图像
    cv2.imshow('AprilTag with Corners', img)

    # 等待按下任意键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取AprilTag图像
image_paths = './dataset/tag36h11_005_img/'
file_list = os.listdir(image_paths)  # 获取文件夹中所有文件的名称
image_list = [os.path.join(image_paths, filename) for filename in file_list]
input_img_shape = cv2.imread(image_list[0]).shape[:2]
label_list = [file_name.replace('.jpg', '.txt').replace('img', 'label') for file_name in image_list]

for image, label in zip(image_list, label_list):
    test_match(image, label)


# test_match('./images/tag36h11_005_000.jpg', 'tag36h11_005_000.txt')

