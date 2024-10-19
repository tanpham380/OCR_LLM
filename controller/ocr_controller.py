import cv2
import numpy as np
from typing import List, Dict, Any

import torch
from app_utils.bbox_fix import crop_and_merge_images, merge_overlapping_bboxes, polygon_to_bbox, remove_duplicate_bboxes
from app_utils.file_handler import load_and_preprocess_image, save_image
from app_utils.logging import get_logger
from app_utils.ocr_package.detection import batch_text_detection
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr

from controller.llm_vison_future import VinternOCRModel 

logger = get_logger(__name__)

import asyncio


class OcrController:
    def __init__(self , model_path: str = "app_utils/weights/Vintern-3B-v1-phase4" , device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.language_list = ["vi" , "en" ]
        
        self.det_processor = TextDect_withRapidocr(text_score = 0.6 , det_use_cuda = True)
        self.vintern_ocr = VinternOCRModel(model_path , device)

    async def scan_ocr(self, image_input: Any) -> str:
        try:
            # Assuming image_input is a list of two images
            if not isinstance(image_input, list) or len(image_input) != 2:
                raise ValueError("image_input must be a list of two images")
            
            # Concatenate the two images vertically
            processed_img, original_img = load_and_preprocess_image(image_input[0])
            processed_img_2, original_img_2 = load_and_preprocess_image(image_input[1])
            
            height_2 = original_img_2.shape[0]
            original_img_2 = original_img_2[:height_2 // 2, :]

            original_img = np.concatenate((original_img, original_img_2), axis=0)

            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            save_image(img_gray)

            det_results = batch_text_detection(img_gray, self.det_processor)

            # Remove first 5 and last 5 elements
            det_results = det_results[5:] 

            bboxes = [polygon_to_bbox(polygon) for polygon in det_results]
            unique_bboxes = remove_duplicate_bboxes(bboxes)
            merged_bboxes = merge_overlapping_bboxes(unique_bboxes)
            merged_image = crop_and_merge_images(original_img, merged_bboxes)
            save_image(merged_image)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vision_model_result = await asyncio.to_thread(self.vintern_ocr.process_image, merged_image)
            if isinstance(vision_model_result, list):
                vision_model_text = f"{str(vision_model_result)}\n\n"
            else:
                vision_model_text = f"{vision_model_result}\n\n"            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return vision_model_text

        except Exception as e:
            raise e