from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from app_utils.file_handler import save_image
from app_utils.util import rotate_image
from config import CORNER_MODEL_PATH, CORNER_MODEL_PATH2

class ImageRectify:
    def __init__(self):
        self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)  
        self.ANGLE_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH2) 

    def load_yolov8_model(self, model_path: str) -> YOLO:
        """Load YOLOv8 model."""
        model = YOLO(model_path)
        model.conf = 0.5
        model.iou = 0.5
        return model

    def process_normal_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process the normal YOLO model to get bounding box."""
        result = self.CORNER_MODEL(image)[0]
        if not result.boxes: 
            return None, None
        class_index = int(result.boxes.cls[0])  
        detected_name = result.names[class_index] 
        boxes = result.boxes.xyxy.cpu().numpy() 
        return boxes, detected_name

    def process_obb_yolo(self, image: np.ndarray) -> Tuple[float]:
        """Process the OBB YOLO model to calculate the angle."""
        results = self.ANGLE_MODEL(image)[0]
        if not hasattr(results, 'obb') or results.obb is None:
            return 0
        obb = results.obb
        data = obb.data.cpu().numpy()
        for i, box in enumerate(data):
            _, _, _, _, rotation_radian = obb.xywhr[i].cpu().numpy()
            angle_deg = np.degrees(rotation_radian)
            return angle_deg

        return 0

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
        """Detect and align the ID card using both YOLO models."""
        boxes, detected_name = self.process_normal_yolo(image)
        if boxes is None:
            return None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]

        angle_deg= self.process_obb_yolo(cropped_image)
                
        rotated_image = rotate_image(cropped_image, angle_deg)

        is_front = detected_name == "front"


        return rotated_image, is_front