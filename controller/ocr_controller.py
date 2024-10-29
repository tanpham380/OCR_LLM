import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

import cv2
import numpy as np
import torch
from functools import lru_cache

from app_utils.bbox_fix import (
    crop_and_merge_images,
    merge_overlapping_bboxes,
    polygon_to_bbox,
    remove_duplicate_bboxes,
)
from app_utils.file_handler import load_and_preprocess_image, save_image
from app_utils.logging import get_logger
from app_utils.ocr_package.detection import batch_text_detection
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from controller.llm_vison_future import VinternOCRModel

logger = get_logger(__name__)

@dataclass
class ImageProcessingConfig:
    """Configuration for image processing parameters"""
    text_score: float = 0.6
    det_use_cuda: bool = True
    crop_ratio: float = 2/3
    crop_ratio_front: float = 1/4

class OcrController:
    def __init__(
        self,
        model_path: str = "app_utils/weights/Vintern-3B-beta",
        config: Optional[ImageProcessingConfig] = None
    ) -> None:
        self.config = config or ImageProcessingConfig()
        self.language_list = ["vi", "en"]
        
        # Initialize detection processor
        self.det_processor = TextDect_withRapidocr(
            text_score=self.config.text_score,
            det_use_cuda=self.config.det_use_cuda
        )
        
        # Initialize OCR models on different devices
        self.devices = self._setup_devices()
        self.ocr_models = self._initialize_ocr_models(model_path)

    @staticmethod
    def _setup_devices() -> List[torch.device]:
        """Setup available CUDA devices"""
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]

    def _initialize_ocr_models(self, model_path: str) -> List[VinternOCRModel]:
        """Initialize OCR models for available devices"""
        return [VinternOCRModel(model_path=model_path, device=device) 
                for device in self.devices]

    @staticmethod
    async def _combine_images(
        img1: Tuple[np.ndarray, np.ndarray],
        img2: Tuple[np.ndarray, np.ndarray],
        crop_ratio: float
    ) -> np.ndarray:
        """Combine two images with optimal layout"""
        _, original_img1 = img1
        _, original_img2 = img2
        
        # Crop second image
        height_2 = original_img2.shape[0]
        original_img2 = original_img2[:int(height_2 * crop_ratio), :]
        
        # Calculate dimensions
        max_width = max(original_img1.shape[1], original_img2.shape[1])
        total_height = original_img1.shape[0] + original_img2.shape[0]
        
        # Create combined image
        combined_img = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        
        # Place first image
        combined_img[:original_img1.shape[0], :original_img1.shape[1]] = original_img1
        
        # Place second image
        x_offset = max_width - original_img2.shape[1]
        y_offset = original_img1.shape[0]
        combined_img[
            y_offset:y_offset + original_img2.shape[0],
            x_offset:x_offset + original_img2.shape[1]
        ] = original_img2
        
        return combined_img

    # async def _process_single_image(self,image: np.ndarray,skip_first_results: bool = False) -> Tuple[np.ndarray, List[List[int]]]:
    #     """Process a single image and return grayscale image and bounding boxes"""
    #     height_image = image.shape[0]
    #     if not skip_first_results:
    #         img_progess = image[:int(height_image * self.config.crop_ratio), :]
    #     else:
    #         img_progess = image[int(height_image * self.config.crop_ratio_front):, :]
    #     # img_gray =  cv2.cvtColor(img_progess, cv2.COLOR_BGR2GRAY) #image

    #     # det_results = await asyncio.to_thread(
    #     #     batch_text_detection, img_gray, self.det_processor
    #     # )
        
    #     # # # Skip first 5 results if needed
    #     # # if skip_first_results:
    #     # #     det_results = det_results[5:]
            
    #     # # # Process detection results
    #     # bboxes = [polygon_to_bbox(polygon) for polygon in det_results]
    #     # unique_bboxes = remove_duplicate_bboxes(bboxes)
    #     # merged_bboxes = merge_overlapping_bboxes(unique_bboxes)
    #     merged_bboxes = None
    #     return img_progess ,  merged_bboxes #None

    async def _process_ocr(self, merged_image: np.ndarray , device_use: int = 0 ) -> str:
        """Process image with OCR model"""
        
        if len(self.devices) < 2:
            logger.warn("Not enough GPUs for dual processing, falling back to single GPU mode")
            device_use = 0  # Use same GPU for both images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        vision_model = self.ocr_models[device_use]
        vision_model_result = await asyncio.to_thread(
            vision_model.process_image, merged_image
        )
        
        vision_model_text = (
            f"{str(vision_model_result)}\n\n"
            if isinstance(vision_model_result, list)
            else f"{vision_model_result}\n\n"
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return vision_model_text

    async def scan_ocr_dual(self, image_input: List[Any]) -> Dict[str, str]:
        """Process two images separately with different interfaces"""
        try:
            if not isinstance(image_input, list) or len(image_input) != 2:
                raise ValueError("image_input must be a list of two images")

            # Load and preprocess images asynchronously
            img1_processed, img1_original = await asyncio.to_thread(
                load_and_preprocess_image, image_input[0]
            )
            img2_processed, img2_original = await asyncio.to_thread(
                load_and_preprocess_image, image_input[1]
            )
            
            
            # # Process images separately with different interfaces
            # img1, bboxes1 = await self._process_single_image(
            #     img1_original,
            #     skip_first_results=True  # Skip first 5 results for image 1
            # )
            # img2, bboxes2 = await self._process_single_image(
            #     img2_original,
            #     skip_first_results=False  # Don't skip results for image 2
            # )
            
            # # Merge and process each image
            # merged_image1 = crop_and_merge_images(img1, bboxes1)
            # merged_image2 = crop_and_merge_images(img2, bboxes2)
            save_image(img1_processed)
            save_image(img2_processed)
            # Process both images with OCR
            results = await asyncio.gather(
                self._process_ocr(img1_processed , device_use = 0),
                self._process_ocr(img2_processed , device_use= 1)
            )
            
            return {
                "image1_result": results[0],
                "image2_result": results[1]
            }

        except Exception as e:
            logger.error(f"Error in scan_ocr_dual: {str(e)}")
            raise

    async def scan_ocr(self, image_input: List[Any]) -> str:
        """Original method for processing combined images"""
        try:
            if not isinstance(image_input, list) or len(image_input) != 2:
                raise ValueError("image_input must be a list of two images")

            # Load and preprocess images asynchronously
            img1 = await asyncio.to_thread(load_and_preprocess_image, image_input[0])
            img2 = await asyncio.to_thread(load_and_preprocess_image, image_input[1])
            
            # Combine images
            original_img = await self._combine_images(img1, img2, self.config.crop_ratio)
            # save_image(original_img)
            
            # Process the combined image
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            det_results = await asyncio.to_thread(
                batch_text_detection, img_gray, self.det_processor
            )
            
            # Skip first 5 results and process
            det_results = det_results[5:]
            bboxes = [polygon_to_bbox(polygon) for polygon in det_results]
            unique_bboxes = remove_duplicate_bboxes(bboxes)
            merged_bboxes = merge_overlapping_bboxes(unique_bboxes)
            
            
            merged_image = crop_and_merge_images(img_gray, merged_bboxes)
            save_image(merged_image)
            # Process with OCR
            return await self._process_ocr(merged_image)

        except Exception as e:
            logger.error(f"Error in scan_ocr: {str(e)}")
            raise