import cv2
import numpy as np

camera_matrix = np.array([
    [1081.100, 0, 563.574],
    [0, 1166.242, 377.802],
    [0, 0, 1]
])

# 从两个图像中检测到的Apriltag的角
points_image1 = [...]  # e.g., [[x1, y1], [x2, y2], ...]
points_image2 = [...]

H, mask = cv2.findHomography(points_image1, points_image2)

object_points = [
    [-s / 2, s / 2, 0],  # Top-left corner
    [s / 2, s / 2, 0],  # Top-right corner
    [s / 2, -s / 2, 0],  # Bottom-right corner
    [-s / 2, -s / 2, 0]  # Bottom-left corner
]

retval, rvec, tvec = cv2.solvePnP(object_points, points_image, camera_matrix, dist_coeffs)
