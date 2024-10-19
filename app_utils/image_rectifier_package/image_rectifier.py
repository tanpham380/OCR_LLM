from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from app_utils.file_handler import save_image
from app_utils.util import rotate_image
from config import CORNER_MODEL_PATH, CORNER_MODEL_PATH2
# CORNER_MODEL_PATH
import torch

import torch
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple
import torch
import numpy as np
from ultralytics import YOLO

class ImageRectify:
    def __init__(self, crop_expansion_factor: float = 0.05):
        self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)  # Replace with your model path
        self.crop_expansion_factor = crop_expansion_factor

    @staticmethod
    def load_yolov8_model(model_path: str) -> YOLO:
        model = YOLO(model_path)
        model.conf = 0.5
        model.iou = 0.5
        model.to('cpu')  # Ensure model runs on CPU
        return model

    def process_normal_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        result = self.CORNER_MODEL(image)[0]
        print(result.boxes)
        if not result.boxes:
            return image , "front"
        class_index = int(result.boxes.cls[0])
        detected_name = result.names[class_index]
        boxes = result.boxes.xyxy.cpu().numpy()
        return boxes, detected_name

    @staticmethod
    def expand_box(box: torch.Tensor, image_shape: Tuple[int, int, int], crop_expansion_factor: float) -> torch.Tensor:
        h, w = image_shape[0], image_shape[1]
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        box_width = x2 - x1
        box_height = y2 - y1

        x1_new = torch.clamp(x1 - box_width * crop_expansion_factor, min=0.0)
        y1_new = torch.clamp(y1 - box_height * crop_expansion_factor, min=0.0)
        x2_new = torch.clamp(x2 + box_width * crop_expansion_factor, max=float(w))
        y2_new = torch.clamp(y2 + box_height * crop_expansion_factor, max=float(h))

        return torch.stack([x1_new, y1_new, x2_new, y2_new])

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
        boxes, detected_name = self.process_normal_yolo(image)
        if len(boxes) == 0:
            return image

        box = torch.from_numpy(boxes[0]).float().cpu()  # Ensure box is on CPU
        expanded_box = self.expand_box(box, image.shape, self.crop_expansion_factor)
        x1, y1, x2, y2 = map(int, expanded_box.tolist())
        cropped_image = image[y1:y2, x1:x2]
        is_front = detected_name == "front"
        return cropped_image, is_front

# class ImageRectify:
#     def __init__(self, crop_expansion_factor: float = 0.05):
#         self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)
#         self.crop_expansion_factor = crop_expansion_factor

#     @staticmethod
#     def load_yolov8_model(model_path: str) -> YOLO:
#         model = YOLO(model_path)
#         model.conf = 0.5
#         model.iou = 0.5
#         return model

#     def process_normal_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
#         result = self.CORNER_MODEL(image)[0]
#         if not result.boxes:
#             return np.array([]), ""
#         class_index = int(result.boxes.cls[0])
#         detected_name = result.names[class_index]
#         boxes = result.boxes.xyxy.cpu().numpy()
#         return boxes, detected_name

#     @staticmethod
#     @torch.jit.script
#     def expand_box(box: torch.Tensor, image_shape: Tuple[int, int, int], crop_expansion_factor: float) -> torch.Tensor:
#         h, w = image_shape[0], image_shape[1]
#         x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

#         box_width = x2 - x1
#         box_height = y2 - y1

#         x1_new = torch.clamp(x1 - box_width * crop_expansion_factor, min=0.0)
#         y1_new = torch.clamp(y1 - box_height * crop_expansion_factor, min=0.0)
#         x2_new = torch.clamp(x2 + box_width * crop_expansion_factor, max=float(w))
#         y2_new = torch.clamp(y2 + box_height * crop_expansion_factor, max=float(h))

#         return torch.stack([x1_new, y1_new, x2_new, y2_new])

#     @torch.no_grad()
#     def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
#         boxes, detected_name = self.process_normal_yolo(image)
#         if len(boxes) == 0:
#             return None

#         box = torch.from_numpy(boxes[0]).float()
#         expanded_box = self.expand_box(box, image.shape, self.crop_expansion_factor)
#         x1, y1, x2, y2 = map(int, expanded_box.tolist())
#         cropped_image = image[y1:y2, x1:x2]
#         is_front = detected_name == "front"
#         return cropped_image, is_front
# class ImageRectify:
#     def __init__(self, crop_expansion_factor: float = 0.05):
#         self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)
#         # self.ANGLE_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH2) 
#         self.crop_expansion_factor = crop_expansion_factor  # New attribute

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

#     def expand_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
#         """Expand the bounding box by a certain factor."""
#         h, w = image_shape[:2]
#         x1, y1, x2, y2 = box

#         # Calculate width and height of the bounding box
#         box_width = x2 - x1
#         box_height = y2 - y1

#         # Expand the bounding box by the factor
#         x1_new = max(0, x1 - int(box_width * self.crop_expansion_factor))
#         y1_new = max(0, y1 - int(box_height * self.crop_expansion_factor))
#         x2_new = min(w, x2 + int(box_width * self.crop_expansion_factor))
#         y2_new = min(h, y2 + int(box_height * self.crop_expansion_factor))

#         return np.array([x1_new, y1_new, x2_new, y2_new])

#     def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
#         """Detect and align the ID card using both YOLO models."""
#         boxes, detected_name = self.process_normal_yolo(image)
#         if boxes is None:
#             return None

#         for box in boxes:
#             expanded_box = self.expand_box(box, image.shape)
#             x1, y1, x2, y2 = map(int, expanded_box)
#             cropped_image = image[y1:y2, x1:x2]
#         # angle_deg = self.process_obb_yolo(cropped_image) 
#         # rotated_image = rotate_image(cropped_image, angle_deg)
#         is_front = detected_name == "front"
#         return cropped_image, is_front
