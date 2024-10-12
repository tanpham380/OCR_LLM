# from typing import Optional, Tuple
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from app_utils.file_handler import save_image
# from app_utils.util import rotate_image
# from config import CORNER_MODEL_PATH, CORNER_MODEL_PATH2

# class ImageRectify:
#     def __init__(self):
#         self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)  
#         # self.ANGLE_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH2) 

#     def load_yolov8_model(self, model_path: str) -> YOLO:
#         """Load YOLOv8 model."""
#         model = YOLO(model_path)
#         model.conf = 0.5
#         model.iou = 0.5
#         return model

#     def process_normal_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
#         """Process the normal YOLO model to get bounding box."""
#         result = self.CORNER_MODEL(image)[0]
#         if not result.boxes: 
#             return image, False
#         class_index = int(result.boxes.cls[0])  
#         detected_name = result.names[class_index] 
#         boxes = result.boxes.xyxy.cpu().numpy() 
#         return boxes, detected_name

#     def detect(self, image: np.ndarray , ) -> Optional[Tuple[np.ndarray, bool]]:
#         """Detect and align the ID card using both YOLO models."""
#         boxes, detected_name = self.process_normal_yolo(image)
#         if boxes is None:
#             return None

#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             cropped_image = image[y1:y2, x1:x2]
#         # angle_deg= self.process_obb_yolo(cropped_image) 
#         # rotated_image = rotate_image(cropped_image, angle_deg)
#         is_front = detected_name == "front"
#         return cropped_image, is_front
    
    
    
    
from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from app_utils.file_handler import save_image
from app_utils.util import rotate_image
from config import CORNER_MODEL_PATH, CORNER_MODEL_PATH2

class ImageRectify:
    def __init__(self, crop_expansion_factor: float = 0.05):
        self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)
        # self.ANGLE_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH2) 
        self.crop_expansion_factor = crop_expansion_factor  # New attribute

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
            return image, False
        class_index = int(result.boxes.cls[0])
        detected_name = result.names[class_index]
        boxes = result.boxes.xyxy.cpu().numpy()
        return boxes, detected_name

    def expand_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Expand the bounding box by a certain factor."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box

        # Calculate width and height of the bounding box
        box_width = x2 - x1
        box_height = y2 - y1

        # Expand the bounding box by the factor
        x1_new = max(0, x1 - int(box_width * self.crop_expansion_factor))
        y1_new = max(0, y1 - int(box_height * self.crop_expansion_factor))
        x2_new = min(w, x2 + int(box_width * self.crop_expansion_factor))
        y2_new = min(h, y2 + int(box_height * self.crop_expansion_factor))

        return np.array([x1_new, y1_new, x2_new, y2_new])

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
        """Detect and align the ID card using both YOLO models."""
        boxes, detected_name = self.process_normal_yolo(image)
        if boxes is None:
            return None

        for box in boxes:
            expanded_box = self.expand_box(box, image.shape)
            x1, y1, x2, y2 = map(int, expanded_box)
            cropped_image = image[y1:y2, x1:x2]
        # angle_deg = self.process_obb_yolo(cropped_image) 
        # rotated_image = rotate_image(cropped_image, angle_deg)
        is_front = detected_name == "front"
        return cropped_image, is_front
