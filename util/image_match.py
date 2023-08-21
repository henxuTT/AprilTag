import cv2
import numpy as np


class ImageMatcher:
    def __init__(self, image1_path: str, image2_path: str):
        self.im1 = cv2.imread(image1_path)
        self.im2 = cv2.imread(image2_path)

        # 初始化特征提取和匹配器
        self.surf = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

        self.matchesMask = None
        self.H = None

    def match_features(self):
        kp1, des1 = self.surf.detectAndCompute(self.im1, None)
        kp2, des2 = self.surf.detectAndCompute(self.im2, None)

        matches = self.bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.9 * n.distance]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        self.H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.matchesMask = mask.ravel().tolist()

        return good, kp1, kp2

    def draw_matches(self):
        good, kp1, kp2 = self.match_features()

        h, w, d = self.im1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, self.H)

        img2 = cv2.polylines(self.im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=self.matchesMask,
                           flags=2)

        im3 = cv2.drawMatches(self.im1, kp1, img2, kp2, good, None, **draw_params)

        cv2.imshow("result", im3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_homography_matrix(self):
        return self.H



