from detector import detect_roi
from util import image_match

model_path = 'model/find_roi.pt'
img_path = 'images/test_4tags2-Camera 74.png'

detector = detect_roi.YOLOv5Detector(model_path)
detections = detector.detect(img_path)

detector.print_results(detections)


#
# matcher = image_match.ImageMatcher("images/tag36h11_005.jpg", "roi.jpg")
# matcher.draw_matches()
# print(matcher.get_homography_matrix())
