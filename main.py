# import cv2
# from detector import detect_roi
# from util import image_match
# import numpy as np
# from pose.pose_estimator import recover_pose_from_homography_opencv
# from util.flatness import are_points_coplanar
# import matplotlib.pyplot as plt
#
# model_path = 'model/find_roi2.pt'
# img_path = 'images/test_result/SMC0cm-Camera 53.png'
# raw_img = cv2.imread(img_path)
#
# K_path = 'camera_parameters.txt'
# with open(K_path, 'r') as file:
#     lines = file.readlines()
# matrix_values = [list(map(float, line.strip().split())) for line in lines]
# K = np.array(matrix_values)
# K = np.array([[1867.834, 0, 949],
#               [0, 1857.395, 529.667],
#               [0, 0, 1]])
# n = np.array([0, 0, 1])
# d = 1000
#
# detector = detect_roi.YOLOv5Detector(model_path)
# detections = detector.detect(img_path)
#
# switch = {
#     '5': 'images/apriltag/tag36h11_05.png',
#     '7': 'images/apriltag/tag36h11_07.png',
#     '36': 'images/apriltag/tag36h11_36.png',
#     '39': 'images/apriltag/tag36h11_39.png',
# }
#
# # detector.print_results(detections)
# # points = []
# # for det in detections:
# #     base_img = cv2.imread(switch[det['class']])
# #     matcher = image_match.ImageMatcher(raw_img, base_img, det['bbox'])
# #     matcher.draw_matches(False)
# #     H = matcher.get_homography_matrix()
# #     R, t = recover_pose_from_homography_opencv(H, K, n, d)
# #     p_world = np.array([0, 0, 50])
# #     p_camera = np.dot(R, p_world) + t
# #     print(p_camera)
# #     points.append(p_camera)
# #
# # p1, p2, p3, p4 = points[:]
# # print(are_points_coplanar(p1, p2, p3, p4, threshold=5))
#
# points = {}
# for det in detections:
#     base_img = cv2.imread(switch[det['class']])
#     matcher = image_match.ImageMatcher(raw_img, base_img, det['bbox'])
#     matcher.draw_matches(False)
#     H = matcher.get_homography_matrix()
#     R, t = recover_pose_from_homography_opencv(H, K, n, d)
#     p_world = np.array([0, 0, 50])
#     p_camera = np.dot(R, p_world) + t
#     print(det['class'])
#     print("R")
#     print(R)
#     print("t")
#     print(t)
#     print(p_camera)
#     points[det['class']] = p_camera  # Store the p_camera value with the key as det['class']
#
# # Assuming you have four detections, you can retrieve the points like this:
# p1 = points.get('5')  # Replace 'class_name_1' with the actual class name
# p2 = points.get('7')
# p3 = points.get('36')
# p4 = points.get('39')
#
# print(are_points_coplanar(p2, p3, p4, p1, threshold=10))
# print(are_points_coplanar(p1, p3, p4, p2, threshold=10))
# print(are_points_coplanar(p1, p2, p4, p3, threshold=10))
# print(are_points_coplanar(p1, p2, p3, p4, threshold=10))
#
#
import cv2
from detector import detect_roi
from util import image_match
import numpy as np
from pose.pose_estimator import recover_pose_from_homography_opencv
from util.flatness import are_points_coplanar
import matplotlib.pyplot as plt

model_path = 'model/find_roi2.pt'
img_path = 'images/test_result/SMC0cm-Camera 53.png'
raw_img = cv2.imread(img_path)

K_path = 'camera_parameters.txt'
with open(K_path, 'r') as file:
    lines = file.readlines()
matrix_values = [list(map(float, line.strip().split())) for line in lines]
K = np.array(matrix_values)
K = np.array([[1867.834, 0, 949],
              [0, 1857.395, 529.667],
              [0, 0, 1]])
n = np.array([0, 0, 1])
d = 1000

detector = detect_roi.YOLOv5Detector(model_path)
detections = detector.detect(img_path)

switch = {
    '5': 'images/apriltag/tag36h11_05.png',
    '7': 'images/apriltag/tag36h11_07.png',
    '36': 'images/apriltag/tag36h11_36.png',
    '39': 'images/apriltag/tag36h11_39.png',
}

points = {}
output_file_path = "output.txt"

with open(output_file_path, 'w') as f:
    for det in detections:
        base_img = cv2.imread(switch[det['class']])
        matcher = image_match.ImageMatcher(raw_img, base_img, det['bbox'])
        matcher.draw_matches(False)
        H = matcher.get_homography_matrix()
        R, t = recover_pose_from_homography_opencv(H, K, n, d)
        p_world = np.array([0, 0, 50])
        p_camera = np.dot(R, p_world) + t
        f.write(f"{det['class']}\n")
        f.write("R\n")
        f.write(f"{R}\n")
        f.write("t\n")
        f.write(f"{t}\n")
        f.write(f"{p_camera}\n")
        points[det['class']] = p_camera  # Store the p_camera value with the key as det['class']

    # Assuming you have four detections, you can retrieve the points like this:
    p1 = points.get('5')  # Replace 'class_name_1' with the actual class name
    p2 = points.get('7')
    p3 = points.get('36')
    p4 = points.get('39')

    f.write(f"{are_points_coplanar(p2, p3, p4, p1, threshold=10)}\n")
    f.write(f"{are_points_coplanar(p1, p3, p4, p2, threshold=10)}\n")
    f.write(f"{are_points_coplanar(p1, p2, p4, p3, threshold=10)}\n")
    f.write(f"{are_points_coplanar(p1, p2, p3, p4, threshold=10)}\n")

