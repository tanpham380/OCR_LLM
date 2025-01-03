import asyncio
from typing import List, Any

import numpy as np
import torch

from app_utils.file_handler import load_and_preprocess_image
from app_utils.logging import get_logger
from controller.vllm_qwen import VLLM_Exes
from prompt import CCCD_FRONT_PROMPT , CCCD_BACK_PROMPT
# from controller.llm_vison_future import VinternOCRModel

logger = get_logger(__name__)



class OcrController:
    def __init__(
        self,
        model_path: str = "erax-ai/EraX-VL-2B-V1.5",
    ) -> None:
        self.language_list = ["vi", "en"]

        self.ocr_model = self._initialize_ocr_models(model_path)



    def _initialize_ocr_models(self, model_path: str) -> VLLM_Exes:
        """Initialize OCR model with optimal GPU configuration"""
        num_gpus = torch.cuda.device_count()
        
        if num_gpus >= 2:
            return VLLM_Exes(
                model_name=model_path,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.95,
                max_seq_len=2048,
                # quantization="awq",
                # dtype="float16",
                swap_space=4,
                max_num_seqs=5
            )
        else:
            return VLLM_Exes.create_optimized("high")
        

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