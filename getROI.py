import pupil_apriltags as apriltag
import cv2
import numpy as np
import os
from util.flatness import are_points_coplanar
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from network.decoder_net import DecoderNet


def original_roi(img, pts):
    x_min = int(np.min(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    x_max = int(np.max(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))

    # 提取矩形区域作为新的ROI
    roi = img[y_min:y_max, x_min:x_max]
    return roi

def transformed_roi(img, pts):
    # 计算ROI的宽度和高度
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))

    # 计算变换矩阵
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.float32(pts), dst_pts)

    # 执行透视变换
    roi = cv2.warpPerspective(img, M, (width, height))
    return roi


img = cv2.imread("images/test_4tags2-Camera 74.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

camera_params = (1867.834, 1857.395, 949, 529.667)  # 你的相机焦距和主点
dist_coeffs = (0.09835, -7.41772, -0.000331945, -0.00191, 129.28)  # 你的相机畸变系数
tag_size = 50


# 创建一个apriltag检测器
detector = apriltag.Detector(families='tag36h11') # windows

# 进行apriltag检测，得到检测到的apriltag的列表
tags = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

points=[]
for tag in tags:
    print(tag.tag_id)
    print("Rotation (pose_r):\n", tag.pose_R)
    print("Translation (pose_t):\n", tag.pose_t)
    p_world = np.array([0, 0, 50])
    p_camera = np.dot(tag.pose_R, p_world) + tag.pose_t.ravel()
    print(p_camera)
    points.append(p_camera)

p1, p2, p3, p4 = points[:]
print(are_points_coplanar(p2, p3, p4, p1, threshold=10))
print(are_points_coplanar(p1, p3, p4, p2, threshold=10))
print(are_points_coplanar(p1, p2, p4, p3, threshold=10))
print(are_points_coplanar(p1, p2, p3, p4, threshold=10))












# copied_img = img.copy()
#
# for tag in tags:
#     cv2.circle(copied_img, tuple(tag.corners[0].astype(int)), 4, (255, 0, 255), 2)  # left-top
#     cv2.circle(copied_img, tuple(tag.corners[1].astype(int)), 4, (255, 0, 255), 2)  # right-top
#     cv2.circle(copied_img, tuple(tag.corners[2].astype(int)), 4, (255, 0, 255), 2)  # right-bottom
#     cv2.circle(copied_img, tuple(tag.corners[3].astype(int)), 4, (255, 0, 255), 2)  # left-bottom
#
# cv2.imshow("out_image", copied_img)
# cv2.waitKey()
#
#
# # 假设你有四个顶点
# # 注意：顶点应按照顺时针或逆时针的顺序排列
# pts = tags[0].corners
# print(pts)
#
# # 提取矩形区域作为新的ROI
# roi = original_roi(img, pts)
# # roi = transformed_roi(img, pts)
#
# # 在新的窗口中显示ROI
# cv2.imshow('ROI', roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# roi = cv2.resize(roi, (256, 256), interpolation = cv2.INTER_LINEAR)
# cv2.imwrite('roi.jpg', roi)
# cv2.imshow('ROI', roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # cv2.imwrite('images/tag36h11_005_000.jpg', roi)


# # 加载模型
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model = DecoderNet()
# model.load_state_dict(torch.load('model_parameters.pth'))
# model = model.to(device)
# model.eval()  # 设置为评估模式
#
#
# h, w = roi.shape[:2]  # Get the height and width of the image
#
# transform = ToTensor()
# image_tensor = transform(roi).unsqueeze(0).to(device)  # Add a batch dimension and convert to Tensor
#
# # Perform inference with model
# with torch.no_grad():
#     _, _, _, location = model(image_tensor)
#
# pixel_coords = location[0].int()  # Use the first batch item and convert to int
#
# # Draw the predicted keypoints onto the image
# for i in range(pixel_coords.shape[0]):
#     cv2.circle(roi, (pixel_coords[i, 0].item(), pixel_coords[i, 1].item()), 3, (0, 0, 255), -1)
#
# cv2.imshow('result', roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()











# # 加载和预处理图像
# image_path = 'dataset/tag_05_image/'
# # image_path1 = 'dataset/tag_05_image/001.png'
# files = os.listdir(image_path)
# for file in files:
#     file_path = os.path.join(image_path, file)
#     image = cv2.imread(file_path)
#
# # print(cv2.imread(image_path1).shape)
#
#     # image = cv2.imread(image_path)
#     # image = cv2.resize(image, (1080, 1080))
#     h, w = image.shape[:2]  # Get the height and width of the image
#
#     transform = ToTensor()
#     image_tensor = transform(image).unsqueeze(0).to(device)  # Add a batch dimension and convert to Tensor
#
#     # Perform inference with model
#     with torch.no_grad():
#         _, _, _, location = model(image_tensor)
#
#     pixel_coords = location[0].int()  # Use the first batch item and convert to int
#
#     # Draw the predicted keypoints onto the image
#     for i in range(pixel_coords.shape[0]):
#         cv2.circle(image, (pixel_coords[i, 0].item(), pixel_coords[i, 1].item()), 3, (0, 0, 255), -1)
#
#     cv2.imshow('result', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()