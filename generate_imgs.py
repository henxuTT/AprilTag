import cv2
import numpy as np
import os

def draw_bbox(img, label):
    h, w = img.shape[:2]

    # 解析YOLO标签
    class_id, x_center, y_center, width, height = label

    # 将边界框的中心坐标、宽度和高度转换为像素值
    x_center *= w
    y_center *= h
    width *= w
    height *= h

    # 计算边界框的左上角和右下角坐标
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # 在图像上画出边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

base_img_path = 'images/office_image'
files = os.listdir(base_img_path)
index = 1
img = cv2.imread('images/tag36h11_005_000.jpg')

x1 = 0
y1 = 0
x2 = 256
y2 = 256
for file in files:
    base_img = cv2.imread(os.path.join(base_img_path, file))
    for i in range(50):
        a1, a2, a3, a4, a5, a6, a7, a8 = np.random.uniform(0.05, 0.7, 8)
        src_points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        dst_points = np.float32(
            [[a1 * x1, a2 * y1], [(1 - a3) * x2, a4 * y1], [(1 - a5) * x2, (1 - a6) * y2], [a7 * x1, (1 - a8) * y2]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 进行透视变换，获取四边形
        dst = cv2.warpPerspective(img, M, (0, 0))

        # 创建一个空白的掩码，大小和你的目标图像一致
        mask = np.zeros((y2, x2), np.uint8)

        # 定义你的四边形的四个顶点
        points = dst_points

        # 将点的坐标数组调整为 (1, 4, 2) 的形状
        pts = points.reshape((-1, 1, 2))

        # 在掩码上填充你的四边形
        cv2.fillConvexPoly(mask, pts.astype(int), 255)

        # # 显示掩码
        # cv2.imshow('Mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(base_img.shape)
        print(dst.shape)

        # 定义随机的插入位置
        startX = np.random.randint(0, base_img.shape[1] - dst.shape[1])
        startY = np.random.randint(0, base_img.shape[0] - dst.shape[0])

        # 创建一个和dst同样大小的掩码，初值为0
        mask = np.zeros((dst.shape[0], dst.shape[1]), dtype=np.uint8)

        # 使用 fillConvexPoly 在掩码上画出你的四边形
        cv2.fillConvexPoly(mask, dst_points.astype(int), (255))

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 将掩码调整为和背景图像同样的大小
        full_mask = base_img.copy()
        full_mask[startY:startY + dst.shape[0], startX:startX + dst.shape[1]] = mask

        # 将dst调整为和背景图像同样的大小
        full_dst = base_img.copy()
        full_dst[startY:startY + dst.shape[0], startX:startX + dst.shape[1]] = dst

        # 在背景图像上加上dst，同时利用掩码只保留四边形部分
        result = cv2.bitwise_and(full_dst, full_mask) + cv2.bitwise_and(base_img, cv2.bitwise_not(full_mask))

        # 显示结果
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 获取四边形的中心点坐标、宽度和高度

        x_min = np.min([dst_points[0][0], dst_points[1][0], dst_points[2][0], dst_points[3][0]])
        x_max = np.max([dst_points[0][0], dst_points[1][0], dst_points[2][0], dst_points[3][0]])
        y_min = np.min([dst_points[0][1], dst_points[1][1], dst_points[2][1], dst_points[3][1]])
        y_max = np.max([dst_points[0][1], dst_points[1][1], dst_points[2][1], dst_points[3][1]])


        centerX = startX + (x_max-x_min) / 2
        centerY = startY + (y_max-y_min) / 2
        width = x_max-x_min
        height = y_max-y_min

        # 转换成相对于图像宽度和高度的比例
        rel_centerX = centerX / result.shape[1]
        rel_centerY = centerY / result.shape[0]
        rel_width = width / result.shape[1]
        rel_height = height / result.shape[0]

        # 定义类别标签，例如，假设四边形的类别为1
        cls = 0

        # 定义YOLOv5标签
        yolo_label = [cls, rel_centerX, rel_centerY, rel_width, rel_height]



        # img_with_bbox = draw_bbox(result, yolo_label)
        # cv2.imshow('Image with Bounding Box', img_with_bbox)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.destroyAllWindows()




        output_image_path = 'VOCdevkit/images/train/005_' + f'{index:03d}.jpg'
        output_label_path = 'VOCdevkit/labels/train/005_' + f'{index:03d}.txt'

        cv2.imwrite(output_image_path, result)

        with open(output_label_path, 'w') as file:
            label_str = ' '.join(map(str, yolo_label))
            file.write(label_str)

        print(index)
        index = index + 1

