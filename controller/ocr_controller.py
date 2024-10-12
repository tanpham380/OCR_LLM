import io
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any

import torch
from app_utils.bbox_fix import is_mrz, merge_overlapping_bboxes, remove_duplicate_bboxes
from app_utils.file_handler import load_and_preprocess_image, save_image
from app_utils.logging import get_logger
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from app_utils.ocr_package.ocr import run_ocr

from app_utils.ocr_package.model.recognition.processor import (
    load_processor as load_rec_processor,
)
from app_utils.ocr_package.model.recognition.model import load_model as load_rec_model
from PIL import Image

from config import VIETOCR_MODEL_PATH
from controller.llm_vison_future import VinternOCRModel , EraxLLMVison

logger = get_logger(__name__)
# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
import asyncio
class OcrController:
    def __init__(self) -> None:
        self.language_list = ["vi" , "en" ]
        self.det_processor = TextDect_withRapidocr(text_score = 0.6 , det_use_cuda = True)
        self.vintern_ocr = VinternOCRModel("/home/gitlab/ocr/app_utils/weights/Vintern-1B-v3")
        # self.vintern_llm = EraxLLMVison("/home/gitlab/ocr/app_utils/weights/EraX-VL-7B-V1")
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        # self.config = Cfg.load_config_from_name('vgg_seq2seq')
        # self.config['weights'] = VIETOCR_MODEL_PATH
        # self.config['cnn']['pretrained'] = False
        # self.config['device'] = 'cpu'
        # self.detector = Predictor(self.config)
    # def get_vintern_llm(self):
    #     return self.vintern_llm


    async def scan_image(self, image_input: Any, methods: List[str] = [ "package_ocr"] , mat_sau: bool = False ) -> Dict[str, Any]:
        """
        Scans the provided image using specified OCR methods.

        Args:
            image_input (Any): Path to the image file, NumPy array, or PIL Image.
            methods (List[str]): List of OCR methods to use.

        Returns:
            Dict[str, Any]: OCR results for each method.

        Raises:
            Exception: If image loading or preprocessing fails.
        """
        processed_img, original_img = load_and_preprocess_image(image_input)
        if processed_img is None:
            raise Exception("Failed to load or preprocess the image.")

        results = {}
        ocr_methods = {
            "package_ocr": self._scan_with_package_ocr,
            # "package_ocr2": self._scan_with_package_ocr2,
        }

        for method in methods:
            if method in ocr_methods:
                try:
                    img_to_use = (
                        original_img if method == "package_ocr" else processed_img
                    )
                    results[method] = await ocr_methods[method](img_to_use , mat_sau)
                except Exception as e:
                    raise Exception(f"Error during scan_image: {e}")
        return results

    async def _scan_with_package_ocr(self, img: np.ndarray, mat_sau: bool = False) -> str:
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_original = Image.fromarray(img_gray)
            # messages = self.vintern_llm.set_prompt_messages(img)
            
            ocr_result = await asyncio.to_thread(
                run_ocr, [img_original], [self.language_list], self.det_processor, self.rec_model, self.rec_processor
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vision_model_result = await asyncio.to_thread(self.vintern_ocr.process_image, img)
            text_lines = ocr_result[0].text_lines
            filtered_text_lines = [line for line in text_lines if line.confidence >= 0.5]
            # if mat_sau:
            #     filtered_text_lines = filtered_text_lines[:len(filtered_text_lines)//2]
            #     filtered_text_lines = [line for line in filtered_text_lines if not is_mrz(line.text)]
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # # self.detector 
            # bboxes = [list(map(int, line.bbox)) for line in filtered_text_lines]
            # image_list = []
            # for bbox in bboxes:
            #     x_min, y_min, x_max, y_max = bbox
            #     cropped_img = img_original.crop((x_min, y_min, x_max, y_max))
            #     image_list.append(cropped_img)

            # Vietocr_result = await asyncio.to_thread(self.detector.predict_batch, image_list)
            # Vietocr_text = f"Trích thông tin Vietocr :\n{Vietocr_result}\n\n"

            formatted_text = "\n".join(line.text for line in filtered_text_lines)
            formatted_section = f"Trích thông tin SuryaOCR:\n{formatted_text}\n\n"
            if isinstance(vision_model_result, list):
                vision_model_text = f"Trích thông tin OCR LLM :\n{str(vision_model_result)}\n\n"
            else:
                vision_model_text = f"Trích thông tin OCR LLM  :\n{vision_model_result}\n\n"
            combined_text = formatted_section  + "\n\n" + vision_model_text   #+ "\n\n" + Vietocr_text 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return combined_text

        except Exception as e:
            raise Exception(f"OCR {e}")
