import cv2
from detector import detect_roi
from util import image_match

model_path = 'model/find_roi.pt'
img_path = 'images/test_4tags2-Camera 74.png'
raw_img = cv2.imread(img_path)

detector = detect_roi.YOLOv5Detector(model_path)
detections = detector.detect(img_path)

switch = {
    5: 'images/apriltag_rightsize/tag36_11_00005.jpg',
    7: 'images/apriltag_rightsize/tag36_11_00007.jpg',
    30: 'images/apriltag_rightsize/tag36_11_00036.jpg',
    '039': 'images/apriltag_rightsize/tag36_11_00039.jpg',
}

# detector.print_results(detections)
for det in detections:
    base_img = cv2.imread(switch[det['class']])

    matcher = image_match.ImageMatcher(raw_img, base_img, det['bbox'])
    # matcher.draw_matches()
    H = matcher.get_homography_matrix()
