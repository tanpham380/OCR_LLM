import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
import torch
import numpy as np
from typing import List
from quart import current_app, g, url_for
from app_utils.file_handler import save_image, scale_up_img
from app_utils.logging import get_logger
from app_utils.prompt import generate_user_context
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import calculate_expiration_date, calculate_sex_from_id, rotate_image
from config import SAVE_IMAGES, TEMP_DIR
from controller.detecter_controller import Detector
from controller.llm_controller import LlmController
from controller.ocr_controller import OcrController

logger = get_logger(__name__)

# Initialize controllers
detector_controller = Detector()
llm_controller = LlmController()
orientation_engine = RapidOrientation()
ocr_controller = OcrController(model_path= "app_utils/weights/Vintern-3B-beta")

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

        # # Chạy các hàm bất đồng bộ scan_image đồng thời và đợi kết quả
        # front_side_ocr, back_side_ocr = await asyncio.gather(
        #     ocr_controller.scan_image(front_result["image_path"], ["package_ocr"]),
        # )
        ocr_text = await ocr_controller.scan_ocr_dual([front_result["image_path"], back_result["image_path"]] )
        combined_ocr_data = {
            "front_side_ocr": ocr_text,
            "front_side_qr": front_result["qr_code_text"],
            "back_side_qr": back_result["qr_code_text"],
            "back_side_ocr": ""
        }
        ocr_result_id = await db_manager.insert_ocr_result(combined_ocr_data)

        await asyncio.gather(
            db_manager.insert_image(ocr_result_id, 'front', front_result["image_path"]),
            db_manager.insert_image(ocr_result_id, 'back', back_result["image_path"])
        )

        context = await asyncio.to_thread(llm_controller.set_user_context, combined_ocr_data)
        await db_manager.insert_user_context(ocr_result_id, context)

        llm_controller.set_model('qwen2.5')
        llm_response = await llm_controller.send_message()
        message_content = clean_message_content(llm_response.get('message', {}).get('content', ''))
        if not message_content.get('date_of_expiration'):
            day_of_birth = message_content.get('day_of_birth', '')
            if day_of_birth:
                expiration_date = calculate_expiration_date(day_of_birth)
                message_content['date_of_expiration'] = expiration_date
        if not message_content.get('nationality'):
            message_content['nationality'] = "Việt Nam"
        if not message_content.get('sex'):
            id_number = message_content.get('id_number', '')
            message_content['sex'] = calculate_sex_from_id(id_number)

        processing_time = time.perf_counter() - start_time
        llm_response_with_time = {
            "llm_response": message_content,
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
        qr_code_text_task = asyncio.to_thread(detector_controller.read_QRcode, image)
        qr_code_text = await asyncio.gather(qr_code_text_task)

        return {
            "qr_code_text": qr_code_text or " ",
            "mat_truoc": mat_truoc,
            "image_path": image_path
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise e

def clean_message_content(message_content: str) -> dict:
    try:
        # First try to extract valid JSON from within backticks
        match = re.search(r'```json\s*(\{.*?\})\s*```', message_content, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            match = re.search(r'(\{.*?\})', message_content, re.DOTALL)
            json_content = match.group(1) if match else "{}" 


        # Fix common issues like using single quotes
        json_content = json_content.replace("'", '"')

        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON content: {e}")
        raise e

