import cv2
import os

# path = 'images/apriltag_origin/'
# files = os.listdir(path)
# for file in files:
#     files_path = os.path.join(path, file)
#     img = cv2.imread(files_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (300, 300))
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
#
#     cv2.imwrite(files_path, img)

path = 'images/office_image/'
files = os.listdir(path)
for file in files:
    files_path = os.path.join(path, file)
    img = cv2.imread(files_path)
    img = cv2.resize(img, (2000, 2000))
    cv2.imshow("img", img)
    cv2.waitKey(0)

    cv2.imwrite(files_path, img)
