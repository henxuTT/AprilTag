import cv2
import numpy as np


def transform_save_2(img, pts_1, pts_2, start_idx, num, output_img_path, output_label_path):
    for i in range(start_idx, num + start_idx):
        x1, y1, x2, y2 = pts_1
        a1, a2, a3, a4, a5, a6, a7, a8 = np.random.uniform(0.05, 0.7, 8)
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
        transformed_points = transformed_points.reshape(-1, 2).astype(int)
        # print(transformed_points)


        x_min = np.min(transformed_points[:, 0])
        y_min = np.min(transformed_points[:, 1])
        x_max = np.max(transformed_points[:, 0])
        y_max = np.max(transformed_points[:, 1])
        crop_img = output_img[y_min:y_max, x_min:x_max]
        mask = np.ones(crop_img.shape, dtype=np.uint8) * 255
        pts = transformed_points - [x_min, y_min]  # 把四边形的坐标转换为相对于bounding box的坐标
        cv2.fillPoly(mask, [pts], (0, 0, 0))
        result = np.where(mask == [0, 0, 0], crop_img, mask)

        # cv2.imshow('crop', result)
        # cv2.waitKey(0)

        output_img_name = output_img_path + f'{i:03d}.jpg'
        output_txt_name = output_label_path + f'{i:03d}.txt'

        np.savetxt(output_txt_name, M)
        cv2.imwrite(output_img_name, result)
        print(i, 'finished')


img = cv2.imread('images/tag36h11_005_000.jpg')
top = bottom = left = right = 50

# 使用copyMakeBorder函数添加白边
border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
# cv2.imshow('border', border)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

x1 = 0
y1 = 0
x2 = img.shape[1]
y2 = img.shape[0]


x3 = x1 + left
y3 = y1 + top
x4 = x2 + left
y4 = y2 + top
pts_2 = np.array([[x3, y3], [x4, y3], [x4, y4], [x3, y4]]).astype(np.float32)
pts_2 = np.array([pts_2])
# print(pts_2)

x2 = x2 + left +right
y2 = y2 + top + bottom
pts_1 = [x1, y1, x2, y2]

output_img_path = 'dataset/STN_dataset/tag36h11_005_img/test/tag36h11_005_'
output_label_path = 'dataset/STN_dataset/tag36h11_005_label/test/tag36h11_005_'


# transform_save_2(border, pts_1, pts_2, 1, 300, output_img_path, output_label_path)
# transform_save(img, pts, points_array,1, 200)
#
# blurred_1 = cv2.GaussianBlur(img, (5, 5), 0)
# transform_save(blurred_1, pts, points_array,201, 200)
#
# blurred_2 = cv2.GaussianBlur(img, (15, 15), 0)
# transform_save(blurred_2, pts, points_array, 401, 200)

try:
    transform_save_2(border, pts_1, pts_2, 1, 10, output_img_path, output_label_path)

    blurred_1 = cv2.GaussianBlur(border, (5, 5), 0)
    # cv2.imshow('blur_1', blurred_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    transform_save_2(blurred_1, pts_1, pts_2, 11, 10, output_img_path, output_label_path)

    blurred_2 = cv2.GaussianBlur(border, (15, 15), 0)
    # cv2.imshow('blur_2', blurred_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    transform_save_2(blurred_2, pts_1, pts_2, 21, 10, output_img_path, output_label_path)
except Exception as e:
    print("An exception occurred: ", e)
