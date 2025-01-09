import asyncio
from typing import List, Any

import numpy as np
import torch

from app_utils.file_handler import load_and_preprocess_image
from app_utils.logging import get_logger
from app_utils.prompt import CCCD_FRONT_PROMPT , CCCD_BACK_PROMPT
from controller.vllm_qwen import VLLM_Exes
# from controller.llm_vison_future import VinternOCRModel

logger = get_logger(__name__)



class OcrController:
    def __init__(
        self,
        model_path: str = "erax-ai/EraX-VL-2B-V1.5",
    ) -> None:
        self.language_list = ["vi", "en"]

        self.ocr_model = VLLM_Exes()


        

    async def scan_ocr(self, image_input: List[Any]) -> str:
        """Process front and back images of ID card"""
        try:
            if not isinstance(image_input, list) or len(image_input) != 2:
                raise ValueError("image_input must be a list of two images")

            # Process front image
            front_img = await asyncio.to_thread(load_and_preprocess_image, image_input[0])
            front_result = await asyncio.to_thread(
                self.ocr_model.generate,
                prompt=CCCD_FRONT_PROMPT,
                image_files=[front_img]
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Process back image 
            back_img = await asyncio.to_thread(load_and_preprocess_image, image_input[1])
            back_result = await asyncio.to_thread(
                self.ocr_model.generate,
                prompt=CCCD_BACK_PROMPT,
                image_files=[back_img]
            )

            final_result = f"{front_result}\n\n{back_result}"
            return final_result

        except Exception as e:
            logger.error(f"Error in scan_ocr: {str(e)}")
            raise