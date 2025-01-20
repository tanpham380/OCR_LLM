from typing import Optional, Tuple
import numpy as np
from ultralytics import YOLO
import torch
from config import CORNER_MODEL_PATH

class ImageRectify:
    def __init__(self, crop_expansion_factor: float = 0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CORNER_MODEL = self.load_yolov8_model(CORNER_MODEL_PATH)
        self.crop_expansion_factor = crop_expansion_factor

    @staticmethod
    def load_yolov8_model(model_path: str) -> YOLO:
        model = YOLO(model_path)
        model.conf = 0.3
        model.iou = 0.3
        model.fuse()  # Fuse model layers for better GPU performance
        return model

    def process_normal_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        result = self.CORNER_MODEL(image, device=self.device)[0]
        if not result.boxes:
            return image, "front"  # Default to front for first image
        class_index = int(result.boxes.cls[0])
        detected_name = result.names[class_index]
        boxes = result.boxes.xyxy.to(self.device)
        return boxes, detected_name

    def expand_box(self, box: torch.Tensor, image_shape: Tuple[int, int, int], crop_expansion_factor: float) -> torch.Tensor:
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
    def detect(self, image: np.ndarray, is_second_image: bool = False) -> Tuple[np.ndarray, bool]:
        """Detect and crop document from image.
        
        Args:
            image: Input image array
            is_second_image: Flag indicating if this is the second image
                
        Returns:
            Tuple of (processed_image, is_front_side)
        """
        boxes, detected_name = self.process_normal_yolo(image)
        
        # Handle no detection case
        if isinstance(boxes, np.ndarray) or (isinstance(boxes, torch.Tensor) and len(boxes) == 0):
            return image, not is_second_image  # First image is front, second is back
            
        # Process detection
        box = boxes[0].float()
        expanded_box = self.expand_box(box, image.shape, self.crop_expansion_factor)
        x1, y1, x2, y2 = map(int, expanded_box.tolist())
        
        cropped_image = image[y1:y2, x1:x2]
        is_front = detected_name == "front"
        
        return cropped_image, is_front