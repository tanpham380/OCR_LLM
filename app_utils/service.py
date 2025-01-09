import asyncio
import json
import os
import re
import time
import torch
from typing import List
from quart import g, url_for
from app_utils.file_handler import crop_back_side, load_image, save_image, scale_up_img
from app_utils.logging import get_logger
from app_utils.prompt import CCCD_FRONT_PROMPT , CCCD_BACK_PROMPT
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
from config import ORIENTATION_MODEL_PATH, SAVE_IMAGES, TEMP_DIR
from controller.detecter_controller import Detector
from controller.vllm_qwen import VLLM_Exes
from controller.llm_controller import LlmController

logger = get_logger(__name__)

detector_controller = Detector()
llm_controller = LlmController()
orientation_engine = RapidOrientation(ORIENTATION_MODEL_PATH)
ocr_controller = VLLM_Exes() 
async def scan(image_paths: List[str]) -> dict:
    try:
        start_time = time.perf_counter()
        db_manager = g.db_manager
        if not db_manager:
            raise RuntimeError("Database manager not initialized.")
        front_result = None
        back_result = None
        mat_sau = False
        for path in image_paths:
            result = await process_image(path, mat_sau)
            if result["mat_truoc"]:
                front_result = result
                mat_sau = True
            else:
                back_result = result

        if not front_result or not back_result:
            raise ValueError("Could not determine front and back images.")
        
        if back_result:
            back_img = load_image(back_result["image_path"])
            cropped_back = crop_back_side(back_img)
       
        ocr_text = ocr_controller.generate_multi([front_result["image_path"], cropped_back])



        processing_time = time.perf_counter() - start_time
        llm_response_with_time = {
            "llm_response": ocr_text,
            "processing_time_seconds": processing_time,
            "mat_truoc": url_for('static', filename=f'images/{os.path.basename(front_result["image_path"])}', _external=True),
            "mat_sau": url_for('static', filename=f'images/{os.path.basename(back_result["image_path"])}', _external=True)
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
        image_path = save_image(image, SAVE_IMAGES, print_path =False)
        # qr_code_text_task = asyncio.to_thread(detector_controller.read_QRcode, image)
        # qr_code_text = await asyncio.gather(qr_code_text_task)

        return {
            # "qr_code_text": qr_code_text or " ",
            "mat_truoc": mat_truoc,
            "image_path": image_path
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise e


