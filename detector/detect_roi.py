import torch
from pathlib import Path
from yolov5.models.yolo import Model


class YOLOv5Detector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        # Load the model definition
        model = Model(cfg='model/yolov5s_apriltag.yaml', ch=3,
                      nc=4)  # Adjust 'path_to_your_model_config.yaml' and num_classes according to your model

        # Load model weights
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])

        self.model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS
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

