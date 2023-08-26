import numpy as np
import cv2
import glob

# 定义棋盘格的角点数 (例如 7x6，意味着有7个内部角点在宽度方向，6个在高度方向)
chessboard_size = (9, 6)

# 设置每个棋盘格的大小，例如30mm
square_size = 8.8

# 生成3D的实际物理坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  # 存储3D点
imgpoints = []  # 存储2D点

# 读取所有图片
images = glob.glob('images/chessboard/*.png')

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到了角点，将它们添加到存储的列表中
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # # 可选：展示检测到的角点
        # cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        # cv2.imshow('Corners', img)
        # cv2.waitKey(100)

cv2.destroyAllWindows()

# 进行相机校准
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix (内参矩阵):\n", mtx)
print("Distortion Coefficients (畸变系数):\n", dist)
