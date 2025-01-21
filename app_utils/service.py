import asyncio
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional, Union
from quart import url_for
from app_utils.file_handler import (
    crop_back_side, load_and_preprocess_image, 
    load_image, save_image, scale_up_img
)
from app_utils.logging import get_logger
from app_utils.prompt import (
    VINTERN_CC_BACK_PROMPT, VINTERN_CC_FRONT_PROMPT,
    VINTERN_CCCD_BACK_PROMPT, VINTERN_CCCD_FRONT_PROMPT
)
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import extract_qr_data, rotate_image
from config import ORIENTATION_MODEL_PATH, SAVE_IMAGES
from controller.detecter_controller import Detector
from controller.openapi_vison import Llm_Vision_Exes
from contextlib import asynccontextmanager

logger = get_logger(__name__)

class CardOCRService:
    def __init__(self):
        self.detector = Detector()
        self.orientation_engine = RapidOrientation(ORIENTATION_MODEL_PATH)
        self.ocr_controller1 = Llm_Vision_Exes(
            api_key="1234",
            api_base="http://172.18.249.58:8000/v1",
            generation_config={"best_of": 1}
        )
        self.ocr_controller2 = Llm_Vision_Exes(
            api_key="1234", 
            api_base="http://172.18.249.58:8001/v1",
            generation_config={"best_of": 1}
        )

    @asynccontextmanager
    async def manage_cuda_memory(self):
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    def validate_ocr_result(self, ocr_result: Dict[str, Any], side: str) -> Dict[str, Any]:
        if not ocr_result or isinstance(ocr_result.get("error"), str):
            logger.error(f"Invalid OCR result for {side} side: {ocr_result}")
            return {}
        return ocr_result
    def get_prompts(self, type_card: str) -> Tuple[str, str]:
        if type_card == "Căn Cước":
            return VINTERN_CC_FRONT_PROMPT, VINTERN_CC_BACK_PROMPT
        return VINTERN_CCCD_FRONT_PROMPT, VINTERN_CCCD_BACK_PROMPT

    def process_ocr(self, image: np.ndarray, prompt: str, controller: Llm_Vision_Exes) -> Dict:
            try:
                result = controller.generate(image, prompt)
                content = result.get("content", "")
                formatted_content = self._format_llm_content(content)
                return formatted_content
            except Exception as e:
                logger.error(f"OCR processing error: {e}")
                return {"error": str(e)}
            
    def _format_llm_content(self, content: Union[str, Dict]) -> Dict:
            try:
                # If content is already a dict, return it
                if isinstance(content, dict):
                    return content
                    
                # If content is string, try to parse as JSON
                if isinstance(content, str):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # If not JSON, wrap in dict
                        return {"text": content}
                        
                # Handle other types
                return {"text": str(content)}
                
            except Exception as e:
                logger.error(f"Error formatting LLM content: {e}")
                return {"error": str(e)}

    async def process_image(self, image_path: str, is_back_side: bool = False) -> Dict[str, Any]:
        try:
            img, _ = load_and_preprocess_image(image_path)
            img = scale_up_img(img, target_size=480)

            async with self.manage_cuda_memory():
                image, is_front_side = await asyncio.to_thread(
                    self.detector.detect, img, is_back_side
                )

            if image is None:
                raise ValueError(f"Document detection failed: {image_path}")

            if is_back_side:
                is_front_side = False

            try:
                orientation_result = await asyncio.to_thread(self.orientation_engine, image)
                orientation_angle = float(orientation_result[0] if isinstance(orientation_result[0], (np.ndarray, list)) else orientation_result[0])
                
                if orientation_angle != 0:
                    image = rotate_image(image, orientation_angle)
            except Exception as e:
                logger.warning(f"Orientation processing failed: {e}")

            qr_code_text = await asyncio.to_thread(self.detector.read_QRcode, image)
            
            if not is_front_side:
                image = crop_back_side(image)

            output_path = save_image(image, SAVE_IMAGES, print_path=False)

            return {
                "qr_code_text": qr_code_text,
                "mat_truoc": is_front_side,
                "image_path": output_path
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise

    async def scan(self, image_paths: List[str]) -> Dict[str, Any]:
        start_time = time.time()

        try:
            if not image_paths or len(image_paths) > 2:
                raise ValueError("Must provide 1-2 images")

            results = []
            for idx, path in enumerate(image_paths):
                result = await self.process_image(path, idx == 1)
                results.append(result)

            # Simplified front/back detection
            front_result = None
            back_result = None
            
            for result in results:
                if result.get("mat_truoc"):
                    front_result = result
                else:
                    back_result = result

            if not front_result or not back_result:
                raise ValueError("Could not determine front/back images")

            qr_data = await extract_qr_data(front_result, back_result)
            type_card = qr_data.get("type_card", "Căn Cước")
            prompt_front, prompt_back = self.get_prompts(type_card)

            front_img = load_image(front_result["image_path"])
            back_img = load_image(back_result["image_path"])

            front_ocr, back_ocr = await asyncio.gather(
                asyncio.to_thread(self.process_ocr, front_img, prompt_front, self.ocr_controller1),
                asyncio.to_thread(self.process_ocr, back_img, prompt_back, self.ocr_controller2)
            )

            # Validate OCR results
            front_ocr = self.validate_ocr_result(front_ocr, "front")
            back_ocr = self.validate_ocr_result(back_ocr, "back")

            ocr_text = {
                **(front_ocr or {}),
                **(back_ocr or {}),
                "qr_code": qr_data.get("qr_data", ""),
                "place_of_issue": qr_data.get("place_of_issue", "Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội"),
                "type_card": type_card
            }

            if not front_ocr and not back_ocr:
                raise ValueError("OCR failed for both front and back sides")

            response = {
                "data": {
                    "llm_response": ocr_text,
                    "mat_truoc": url_for('static', filename=f'images/{Path(front_result["image_path"]).name}', _external=True),
                    "mat_sau": url_for('static', filename=f'images/{Path(back_result["image_path"]).name}', _external=True),
                    "processing_time_s": round(time.time() - start_time, 2)
                }
            }

            return response

        except Exception as e:
            logger.error(f"Scan error: {e}")
            raise

# Initialize service
# ocr_service = CardOCRService()