import cv2
import numpy as np


def transform_save(img, pts, points_array, start_idx, num):
    x1, y1, x2, y2 = pts
    for i in range(start_idx, num+start_idx):
        # 随机生成四边形的顶点坐标 ai
        a1, a2, a3, a4, a5, a6, a7, a8 = np.random.uniform(0.05, 0.7, 8)

        # 定义原始图像中ROI的四个顶点
        src_points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        # 定义目标图像中的四个顶点，映射到随机的四边形图像块
        # dst_points = np.float32([[a1, a2], [1-a3, a4], [1-a5, 1-a6], [a7, 1-a8]])
        dst_points = np.float32(
            [[a1 * x1, a2 * y1], [(1 - a3) * x2, a4 * y1], [(1 - a5) * x2, (1 - a6) * y2], [a7 * x1, (1 - a8) * y2]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        print(M)

        # 执行透视变换，将ROI映射到四边形图像块
        output_img = cv2.warpPerspective(img, M, (0, 0))
        cv2.imshow('transform', output_img)
        cv2.waitKey(0)

        output_img_name = './dataset/tag36h11_005_img/tag36h11_005_' + f'{i:03d}.jpg'
        output_txt_name = './dataset/tag36h11_005_label/tag36h11_005_' + f'{i:03d}.txt'

        # cv2.imwrite(output_img_name, output_img)

        # # 计算特征点的投影变换
        # transformed_points = cv2.perspectiveTransform(points_array, M)
        #
        # # 将浮点数类型的坐标转换为整数
        # transformed_points = transformed_points.reshape(-1, 2).astype(int)

        # # 保存特征点投影变换后的坐标到txt文件
        # with open(output_txt_name, 'w+') as f:
        #     for point in transformed_points:
        #         x, y = point
        #         f.write(f'{x} {y}\n')
        # print(i, 'finish')

        # # 在图像上绘制特征点
        # for point in transformed_points:
        #     x, y = point.astype(int)
        #     cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)
        #
        #
        # # 显示结果图像
        # cv2.imshow('Output Image', output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()





# 读取原始图像和ROI区域
img = cv2.imread('./images/tag36h11_005_000.jpg')
print(img.shape)

x1 = 0
y1 = 0
x2 = img.shape[1]
y2 = img.shape[0]
pts = [x1, y1, x2, y2]

# 读取txt文件
txt_file = 'tag36h11_005_000.txt'
points = []

with open(txt_file, 'r') as f:
    lines = f.readlines()

# 提取特征点坐标
for line in lines:
    x, y = line.strip().split()
    points.append((int(x), int(y)))

# 将坐标数组转换为NumPy数组并调整维度
points_array = np.array(points).reshape(-1, 1, 2).astype(np.float32)


# transform_save(img, pts, points_array,1, 200)
#
# blurred_1 = cv2.GaussianBlur(img, (5, 5), 0)
# transform_save(blurred_1, pts, points_array,201, 200)
#
# blurred_2 = cv2.GaussianBlur(img, (15, 15), 0)
# transform_save(blurred_2, pts, points_array, 401, 200)

try:
    transform_save(img, pts, points_array,1, 200)
    blurred_1 = cv2.GaussianBlur(img, (5, 5), 0)
    transform_save(blurred_1, pts, points_array,201, 200)
    blurred_2 = cv2.GaussianBlur(img, (15, 15), 0)
    transform_save(blurred_2, pts, points_array, 401, 200)
except Exception as e:
    print("An exception occurred: ", e)




