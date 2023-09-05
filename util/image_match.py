import cv2
import numpy as np

class ImageMatcher:
    def __init__(self, raw_img, origin_img, bbox):
        self.raw_img = raw_img
        self.origin_img = origin_img

        padding = 50
        self.x_min = max(int(bbox[0]) - padding, 0)
        self.y_min = max(int(bbox[1]) - padding, 0)
        self.x_max = min(int(bbox[2]) + padding, self.raw_img.shape[1])
        self.y_max = min(int(bbox[3]) + padding, self.raw_img.shape[0])
        self.roi = raw_img[self.y_min:self.y_max, self.x_min:self.x_max]

        self.surf = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

        self.matchesMask = None
        self.H = None

    def match_features(self):
        kp1, des1 = self.surf.detectAndCompute(self.origin_img, None)
        kp2, des2 = self.surf.detectAndCompute(self.roi, None)

        for keypoint in kp2:
            keypoint.pt = (keypoint.pt[0] + self.x_min, keypoint.pt[1] + self.y_min)

        matches = self.bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.9 * n.distance]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        self.H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.matchesMask = mask.ravel().tolist()

        return good, kp1, kp2

    def draw_matches(self, flag=False):
        good, kp1, kp2 = self.match_features()

        h, w, d = self.raw_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, self.H)

        matched_img = self.raw_img.copy()
        matched_img = cv2.polylines(matched_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=self.matchesMask,
                           flags=2)

        im3 = cv2.drawMatches(self.origin_img, kp1, self.raw_img, kp2, good, None, **draw_params)
        im3 = cv2.resize(im3, (1440, 405))

        if flag:
            cv2.imshow("result", im3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("feature match.jpg", im3)

    def get_homography_matrix(self):
        return self.H
