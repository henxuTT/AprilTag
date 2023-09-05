import numpy as np
import cv2

# # 已知的相机内参矩阵 K
# K = np.array([[1867.834, 0, 949],
#               [0, 1857.395, 529.667],
#               [0, 0, 1]])
#
# # 已知的单应矩阵 H
# H = np.array([[0.38879, 0.015976, 461.13],
#               [0.0027857, 0.46286, 1412.4],
#               [-0.000032737, 0.00008512, 1]])  # 用实际的数据替换三个点号
#
# n = np.array([0, 0, 1])  # 场景平面的法线
# d = 1000  # 场景平面到相机中心的距离

def recover_pose_from_homography_opencv(H, K, n, d):
    # Decompose the homography matrix
    retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    # Select the best solution based on the given plane normal and distance
    for i in range(len(rotations)):
        # Check if the normal vector is close to the given normal
        if np.dot(normals[i].ravel(), n) > 0.9:  # Adjust the threshold (0.9) if needed
            # Adjust the translation vector using the given distance
            t = translations[i].ravel() * d / np.dot(normals[i].ravel(), translations[i].ravel())
            return rotations[i], t.ravel()

    # If no suitable solution is found, return None for both R and t
    return None, None