import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CameraVisualizer:
    def __init__(self, depth, rotation_matrix, translation_vector):
        self.depth = depth
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector

    def calculate_new_point(self):
        P1 = np.array([0, 0, self.depth]).reshape(3, 1)
        P2 = np.dot(self.rotation_matrix, P1) + self.translation_vector.reshape(3, 1)
        return P2

    def plot(self):
        P2 = self.calculate_new_point()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制第一位置的相机坐标轴
        ax.quiver(0, 0, 0, 1, 0, 0, length=20, normalize=True, color='r')
        ax.quiver(0, 0, 0, 0, 1, 0, length=20, normalize=True, color='g')
        ax.quiver(0, 0, 0, 0, 0, 1, length=20, normalize=True, color='b')

        # 绘制第二位置的相机坐标轴
        ax.quiver(self.translation_vector[0], self.translation_vector[1], self.translation_vector[2],
                  self.rotation_matrix[0, 0], self.rotation_matrix[1, 0], self.rotation_matrix[2, 0],
                  length=20, normalize=True, color='r')
        ax.quiver(self.translation_vector[0], self.translation_vector[1], self.translation_vector[2],
                  self.rotation_matrix[0, 1], self.rotation_matrix[1, 1], self.rotation_matrix[2, 1],
                  length=20, normalize=True, color='g')
        ax.quiver(self.translation_vector[0], self.translation_vector[1], self.translation_vector[2],
                  self.rotation_matrix[0, 2], self.rotation_matrix[1, 2], self.rotation_matrix[2, 2],
                  length=20, normalize=True, color='b')

        # 绘制点
        ax.scatter(0, 0, self.depth, c='k', marker='o')
        ax.scatter(P2[0], P2[1], P2[2], c='k', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Camera Poses and Point Projection")

        plt.show()
