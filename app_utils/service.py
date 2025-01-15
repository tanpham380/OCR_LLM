import asyncio
import os
import time
import torch
from typing import List
from quart import g, url_for
from app_utils.file_handler import crop_back_side, load_image, merge_images_vertical, save_image, scale_up_img
from app_utils.logging import get_logger
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import extract_qr_data, rotate_image
from config import ORIENTATION_MODEL_PATH, SAVE_IMAGES, TEMP_DIR
from controller.detecter_controller import Detector
from controller.openapi_vison import Llm_Vision_Exes
# from controller.vllm_qwen_old import VLLM_Exes
from controller.llm_controller import LlmController

logger = get_logger(__name__)

detector_controller = Detector()
llm_controller = LlmController()
orientation_engine = RapidOrientation(ORIENTATION_MODEL_PATH)
ocr_controller = Llm_Vision_Exes(
    api_key="1", 
api_base="http://172.18.249.58:8000/v1")


async def scan(image_paths: List[str]) -> dict:
    start_time = time.time()

    try:
        # Process both images concurrently
        results = await asyncio.gather(*[process_image(path) for path in image_paths])
        
        # Classify front and back results
        front_result = next((r for r in results if r["mat_truoc"]), None)
        back_result = next((r for r in results if not r["mat_truoc"]), None)

        if not front_result or not back_result:
            raise ValueError("Could not determine front and back images.")
            
        # Extract QR data
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
        ocr_response["qr_code"] = qr_data
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

async def process_image(image_path: str , mat_sau = False) -> dict:
    try:
        image, mat_truoc = await asyncio.to_thread(detector_controller.detect, image_path)
        if image is None:
            raise ValueError(f"Failed to detect image at {image_path}.")
        if mat_sau :
            mat_truoc = False
        orientation_res, _ = await asyncio.to_thread(orientation_engine, image)
        orientation_res = float(orientation_res)
        if orientation_res != 0:
            image = rotate_image(image, orientation_res)
        image = scale_up_img(image, 480)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        qr_code_text_task = asyncio.to_thread(detector_controller.read_QRcode, image)
        qr_code_text = await asyncio.gather(qr_code_text_task)
        if not mat_truoc:
            image = crop_back_side(image)
        image_path = save_image(image, SAVE_IMAGES, print_path =False)
        return {
            "qr_code_text": qr_code_text or " ",
            "mat_truoc": mat_truoc,
            "image_path": image_path
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise e


