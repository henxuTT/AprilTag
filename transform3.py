import cv2
import numpy as np


def transform_save_3(img, pts_1, pts_2, start_idx, num, output_img_path, output_label_path):
    for i in range(start_idx, num + start_idx):
        x1, y1, x2, y2 = pts_1
        a1, a2, a3, a4, a5, a6, a7, a8 = np.random.uniform(0.05, 0.6, 8)
        src_points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        dst_points = np.float32(
            [[a1 * x1, a2 * y1], [(1 - a3) * x2, a4 * y1], [(1 - a5) * x2, (1 - a6) * y2], [a7 * x1, (1 - a8) * y2]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 执行透视变换，将ROI映射到四边形图像块
        output_img = cv2.warpPerspective(img, M, (0, 0))
        # cv2.imshow('transform', output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        transformed_points = cv2.perspectiveTransform(pts_2, M)
        transformed_points = transformed_points.reshape(-1, 2)

        # x_min = np.min(dst_points[:, 0]).astype(int)
        # y_min = np.min(dst_points[:, 1]).astype(int)
        # x_max = np.max(dst_points[:, 0]).astype(int)
        # y_max = np.max(dst_points[:, 1]).astype(int)
        # crop_img = output_img[y_min:y_max, x_min:x_max]
        # mask = np.ones(crop_img.shape, dtype=np.uint8) * 255
        # pts = transformed_points - [x_min, y_min]  # 把四边形的坐标转换为相对于bounding box的坐标
        # cv2.fillPoly(mask, [pts], (0, 0, 0))
        # result = np.where(mask == [0, 0, 0], crop_img, mask)

        # # 在图像上绘制特征点
        # for point in transformed_points:
        #     x, y = point.astype(int)
        #     cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)
        # # 显示结果图像
        # cv2.imshow('Output Image', output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        output_img_name = output_img_path + f'{i:03d}.jpg'
        output_txt_name = output_label_path + f'{i:03d}.txt'

        cv2.imwrite(output_img_name, output_img)

        # 保存特征点投影变换后的坐标到txt文件
        with open(output_txt_name, 'w+') as f:
            for point in transformed_points:
                x, y = point
                f.write(f'{x} {y}\n')

        print(i, 'finish')










img = cv2.imread('images/tag36h11_005.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
# cv2.imshow("1", img)
# cv2.waitKey(0)
# top = bottom = left = right = 50

# 使用copyMakeBorder函数添加白边
# border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
# cv2.imshow('border', border)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

x1 = 0
y1 = 0
x2 = img.shape[1]
y2 = img.shape[0]
pts_1 = [x1, y1, x2, y2]

# 读取txt文件
txt_file = 'tag36h11_005_000.txt'
points = []

with open(txt_file, 'r') as f:
    lines = f.readlines()

# 提取特征点坐标
for line in lines:
    x, y = line.strip().split()
    points.append((x, y))

points_array = np.array(points).reshape(-1, 1, 2).astype(np.float32)

# for point in points:
#     x, y = point
#     cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
#     # 显示结果图像
#     cv2.imshow('Output Image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




output_img_path = 'dataset/stn_mobilenet/tag36h11_005_img/train/tag36h11_005_'
output_label_path = 'dataset/stn_mobilenet/tag36h11_005_label/train/tag36h11_005_'

output_test_img_path = 'dataset/stn_mobilenet/tag36h11_005_img/test/tag36h11_005_'
output_test_label_path = 'dataset/stn_mobilenet/tag36h11_005_label/test/tag36h11_005_'

try:
    transform_save_3(img, pts_1, points_array, 1, 500, output_img_path, output_label_path)

    blurred_1 = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow('blur_1', blurred_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    transform_save_3(blurred_1, pts_1, points_array, 501, 500, output_img_path, output_label_path)

    blurred_2 = cv2.GaussianBlur(img, (15, 15), 0)
    # cv2.imshow('blur_2', blurred_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    transform_save_3(blurred_2, pts_1, points_array, 1001, 500, output_img_path, output_label_path)
except Exception as e:
    print("An exception occurred: ", e)

# transform_save_3(border, pts_1, points_array, 1, 5, output_test_img_path, output_test_label_path)
# blurred_1 = cv2.GaussianBlur(border, (5, 5), 0)
# transform_save_3(blurred_1, pts_1, points_array, 6, 5, output_test_img_path, output_test_label_path)