import asyncio
import os
import time
import numpy as np
import torch
from typing import Any, Dict, List
from quart import g, url_for
from app_utils.file_handler import crop_back_side, load_and_preprocess_image, load_image, merge_images_vertical, save_image, scale_up_img
from app_utils.logging import get_logger
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import extract_qr_data, rotate_image
from config import ORIENTATION_MODEL_PATH, SAVE_IMAGES, TEMP_DIR
from controller.detecter_controller import Detector
from controller.openapi_vison import Llm_Vision_Exes
# from controller.vllm_qwen_old import VLLM_Exes
from controller.llm_controller import LlmController
from contextlib import asynccontextmanager

logger = get_logger(__name__)

detector_controller = Detector()
llm_controller = LlmController()
orientation_engine = RapidOrientation(ORIENTATION_MODEL_PATH)
# ocr_controller = Llm_Vision_Exes(
#     api_key="1", 
# api_base="http://127.0.0.1:2242/v1" ,

#     generation_config = {
#         "best_of": 2
#     }

# )
@asynccontextmanager
async def manage_cuda_memory():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

ocr_controller = Llm_Vision_Exes(
    api_key="1234", 
    api_base="http://172.18.249.58:8000/v1",
    generation_config = {
        "max_tokens": 512,
        "best_of": 1
    }
)
# 127.0.0.1:2242


async def scan(image_paths: List[str]) -> Dict[str, Any]:
    start_time = time.time()

    try:
        if not image_paths:
            raise ValueError("No image paths provided")
        
        if len(image_paths) > 2:
            raise ValueError("Maximum 2 images can be processed")

        async def process_with_side(path: str, index: int) -> Dict[str, Any]:
            is_back_side = index == 1
            return await process_image(path, is_back_side)

        # Process images with their corresponding side information
        tasks = [
            process_with_side(path, idx) 
            for idx, path in enumerate(image_paths)
        ]
        
        results = await asyncio.gather(*tasks)
        
        front_result = next((r for r in results if r["mat_truoc"]), None)
        back_result = next((r for r in results if not r["mat_truoc"]), None)


        if not front_result or not back_result:
            raise ValueError("Could not determine front and back images.")
            
        # Extract QR data
        print(front_result)
        print("---")
        print(back_result)
        qr_data = await extract_qr_data(front_result, back_result)
        
        # Process images
        back_img = load_image(back_result["image_path"])
        front_img = load_image(front_result["image_path"])
        
        if front_img is None:
            raise ValueError("Failed to load front image")
            
        merged_img = merge_images_vertical(front_img, back_img)
        if merged_img is None:
            raise ValueError("Failed to merge images")
        save_image(merged_img , TEMP_DIR)
        ocr_text = ocr_controller.generate(merged_img)
        ocr_response = ocr_text.get("content", {})
        if isinstance(ocr_response, str):
            import json
            try:
                ocr_response = json.loads(ocr_response)
            except:
                ocr_response = {}
        ocr_response["qr_code"] = qr_data.get("qr_data" , "")
        ocr_response["place_of_issue"] = qr_data.get("place_of_issue")
        ocr_response["type_card"] = qr_data.get("type_card")
        ocr_text["content"] = ocr_response
        
        llm_response_with_time = {
            "llm_response": ocr_text,
            "mat_truoc": url_for('static', filename=f'images/{os.path.basename(front_result["image_path"])}', _external=True),
            "mat_sau": url_for('static', filename=f'images/{os.path.basename(back_result["image_path"])}', _external=True),
            "processing_time_s": round(time.time() - start_time, 2)
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return llm_response_with_time

    except Exception as e:
        logger.error(f"An error occurred during the scanning process: {e}")
        raise e




async def process_image(image_path: str, is_back_side: bool = False) -> Dict[str, any]:
    """Process an image and extract QR code information."""
    try:
        # Load and preprocess image
        img, _ = load_and_preprocess_image(image_path)
        img = scale_up_img(img, target_size=480)

        # Detect document in image
        async with manage_cuda_memory():
            image, is_front_side = await asyncio.to_thread(
                detector_controller.detect, 
                img, 
                is_back_side
            )
            # image, is_front_side = await asyncio.to_thread(detector_controller.detect, img)
            
        if image is None:
            raise ValueError(f"Failed to detect document in image: {image_path}")

        # Override side detection if explicitly processing back side
        if is_back_side:
            is_front_side = False

        # Handle image orientation with proper type checking
        try:
            orientation_result = await asyncio.to_thread(orientation_engine, image)
            orientation_angle = orientation_result[0]
            if isinstance(orientation_angle, (np.ndarray, list)):
                orientation_angle = orientation_angle[0]
            orientation_angle = float(orientation_angle)
            
            if orientation_angle != 0:
                image = rotate_image(image, orientation_angle)
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"Failed to process orientation, using original image: {e}")
            orientation_angle = 0

        # Read QR code
        qr_code_text = await asyncio.to_thread(detector_controller.read_QRcode, image)
        # Process back side if needed
        if not is_front_side:
            image = crop_back_side(image)

        # Save processed image
        output_path = save_image(image, SAVE_IMAGES, print_path=False)

        return {
            "qr_code_text": qr_code_text,
            "mat_truoc": is_front_side,
            "image_path": output_path
        }

    except ValueError as e:
        logger.error(f"Validation error processing image {image_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise 
