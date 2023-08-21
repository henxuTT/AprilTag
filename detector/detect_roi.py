import torch
from pathlib import Path


class YOLOv5Detector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, autoshape=True)
        self.model.eval()

    def detect(self, img_path: str):
        results = self.model(img_path)
        labels = results.pred[0][:, -1].int().tolist()
        boxes = results.pred[0][:, :-1].tolist()

        names = results.names
        class_names = [names[i] for i in labels]

        detections = []
        for box, class_name in zip(boxes, class_names):
            x_min, y_min, x_max, y_max, confidence = box
            detections.append({
                "class": class_name,
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": confidence
            })

        return detections

    def print_results(self, detections):
        for det in detections:
            print(f"Class: {det['class']}, BBox: {det['bbox']}, Confidence: {det['confidence']}")


