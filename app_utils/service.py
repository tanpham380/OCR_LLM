import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import re
from typing import List
import numpy as np
import torch
from app_utils.file_handler import save_image, scale_up_img
from app_utils.logging import get_logger
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import calculate_expiration_date, rotate_image
from config import SAVE_IMAGES
from controller.detecter_controller import Detector
from controller.llm_controller import LlmController
import asyncio
import json
import time
import re
import torch

from quart import current_app, g

logger = get_logger(__name__)

# Initialize controllers
detector_controller = Detector()
llm_controller = LlmController()
orientation_engine = RapidOrientation()


async def detect_and_preprocess_image(image_path: str, mat_sau: bool) -> np.ndarray:
    """
    Detects the image and preprocesses it for further processing.

    Args:
        image_path (str): The path to the image file.
        mat_sau (bool): Indicates if the card orientation should be checked.

    Returns:
        np.ndarray: The processed image.

    Raises:
        Exception: If an error occurs during detection or preprocessing.
    """
    try:
        # Detect the card in the image
        
        image = await detector_controller.detect(image_path)
        if image is None:
            raise Exception("Failed to detect image.")
        orientation_res, _ = orientation_engine(image)
        print(orientation_res)
        if orientation_res != 0 :
            image = rotate_image(image, orientation_res)
        # if mat_sau:
        #     up_down_check = await detector_controller.detect_card_orientation(image)
        # else:
        #     up_down_check = detector_controller.detect_face_orientation(image)
        # if up_down_check:
        #     image = rotate_image(image, 180)
        return image
    except Exception as e:
        logger.error(f"Error during image detection and preprocessing: {e}")
        raise e

def read_qr_code(image: np.ndarray) -> str:
    """
    Reads the QR code from the image.

    Args:
        image (np.ndarray): The image in which QR code is to be read.

    Returns:
        str: The text data from the QR code or an empty string if not found.

    Raises:
        Exception: If an error occurs while reading the QR code.
    """
    try:
        return detector_controller.read_QRcode(image) or ""
    except Exception as e:
        logger.error(f"Error reading QR Code: {e}")
        raise e

async def perform_ocr(image: np.ndarray) -> dict:
    """
    Performs OCR on the provided image using the detector.

    Args:
        image (np.ndarray): The image on which OCR is to be performed.

    Returns:
        dict: The OCR results.

    Raises:
        Exception: If an error occurs during OCR.
    """
    try:
        return await detector_controller.get_ocr().scan_image(image, ["package_ocr"])
    except Exception as e:
        logger.error(f"Error during OCR scan: {e}")
        raise e
async def process_with_llm(dict_ocr: dict) -> dict:
    """
    Processes the OCR data with an LLM and returns the response as a dictionary.

    Args:
        dict_ocr (dict): The OCR results and QR code data.

    Returns:
        dict: The response from LLM in dictionary format.

    Raises:
        Exception: If an error occurs during LLM processing.
    """
    try:
        
        llm_controller.set_user_context(dict_ocr)
        response = await llm_controller.send_message()
        
        # Extract the message content from the response
        message_content = response.get('message', {}).get('content', '')
        
        # Convert the message content to JSON
        if isinstance(message_content, str):
            try:
                # Try parsing the content if it's a JSON string
                message_json = json.loads(message_content)
            except json.JSONDecodeError:
                # If parsing fails, treat it as plain text
                message_json = {"error": "Failed to parse JSON content"}
        else:
            message_json = message_content
        
        return message_json
    except Exception as e:
        logger.error(f"Error during LLM processing: {e}")
        raise e

async def process_with_llm_custom(system_prompt: str = "", user_prompt: str = '', custom_image: str = '') -> str:
    """
    Processes custom data with an LLM.

    Args:
        system_prompt (str): The system prompt for LLM.
        user_prompt (str): The user prompt for LLM.
        custom_image (str): Custom image data.

    Returns:
        str: The formatted response string from LLM.

    Raises:
        ValueError: If user_prompt is None.
        Exception: If an error occurs during LLM processing.
    """
    if user_prompt is None:
        raise ValueError("User prompt cannot be None.")
    try:
        response = await llm_controller.send_custom_message(system_prompt, user_prompt, custom_image)
        result = response.get('message', {}).get('content', '')
        return result
    except Exception as e:
        logger.error(f"Error during custom LLM processing: {e}")
        raise e
    

async def scan(image_paths: List[str]) -> dict:
    """
    Scans the ID card images, processes the text with LLM,
    and returns the corrected information with processing time.
    """
    try:
        start_time = time.perf_counter()
        db_manager = g.db_manager
        if not db_manager:
            raise RuntimeError("Database manager not initialized.")

        # Process both images concurrently
        ocr_tasks = [process_image(path) for path in image_paths]
        
        ocr_results = await asyncio.gather(*ocr_tasks)

        # Determine front and back images
        front_result = next((res for res in ocr_results if res["mat_truoc"]), None)
        back_result = next((res for res in ocr_results if not res["mat_truoc"]), None)

        if not front_result or not back_result:
            raise ValueError("Could not determine front and back images.")
        combined_ocr_data = {
            "front_side_ocr": front_result["ocr_text"],
            "front_side_qr": front_result["qr_code_text"],
            "back_side_ocr": back_result["ocr_text"],
            "back_side_qr": back_result["qr_code_text"]
        }


        ocr_result_id = await db_manager.insert_ocr_result(combined_ocr_data)

        await asyncio.gather(
            db_manager.insert_image(ocr_result_id, 'front', front_result["image_path"]),
            db_manager.insert_image(ocr_result_id, 'back', back_result["image_path"])
        )

        # LLM processing
        context = await asyncio.to_thread(llm_controller.set_user_context, combined_ocr_data)
        await db_manager.insert_user_context(ocr_result_id, context)

        llm_controller.set_model('qwen2.5')
        # llm_controller.set_model('llama3.2:3b')

        
        llm_response = await llm_controller.send_message()
        print(llm_response)
        message_content = clean_message_content(llm_response.get('message', {}).get('content', ''))
        # message_json = json.loads(message_content)
        if message_content.get('date_of_expiration', '') == '':
            day_of_birth = message_content.get  ('day_of_birth', '')
            if day_of_birth:
                expiration_date = calculate_expiration_date(day_of_birth)
                message_content['date_of_expiration'] = expiration_date
        processing_time = time.perf_counter() - start_time
        llm_response_with_time = {
            "llm_response": message_content,
            "processing_time_seconds": processing_time
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await db_manager.insert_scan_result(llm_response_with_time)

        return llm_response_with_time

    except Exception as e:
        logger.error(f"An error occurred during the scanning process: {e}")
        raise e

async def process_image(image_path: str) -> dict:
    """
    Processes a single ID card image for OCR, QR code detection, and orientation correction.
    """
    try:
        image, mat_truoc = detector_controller.detect(image_path)
        if image is None:
            raise ValueError(f"Failed to detect image at {image_path}.")        
        orientation_res, _ = orientation_engine(image)
        orientation_res = float(orientation_res)
        if orientation_res != 0 :
            image = rotate_image(image, orientation_res)
        image = scale_up_img(image, 512)
        image_path = save_image(image, SAVE_IMAGES ,print_path = False )
        qr_code_text_task = asyncio.to_thread(detector_controller.read_QRcode, image)
        ocr_text_task = asyncio.to_thread(detector_controller.get_ocr().scan_image, image, ["package_ocr"])
        qr_code_text, ocr_text = await asyncio.gather(qr_code_text_task, ocr_text_task)

        return {
            "ocr_text":  await ocr_text,
            "qr_code_text":  qr_code_text or " ",
            "mat_truoc": mat_truoc,
            "image_path": image_path 
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise



def clean_message_content(message_content: str) -> dict:
    """
    Extracts the JSON content from the message_content, cleans it, and returns it as a dictionary.
    """
    # Use regex to extract content between ```json and ```
    match = re.search(r'```json\s*(\{.*?\})\s*```', message_content, re.DOTALL)
    if match:
        json_content = match.group(1)
    else:
        # If not found, try to find any JSON object in the message_content
        match = re.search(r'(\{.*?\})', message_content, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            # No JSON content found
            print("Không tìm thấy nội dung JSON trong tin nhắn.")
            return {}
    
    # Try loading the json_content
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Lỗi khi phân tích chuỗi JSON: {e}")
        raise e  # Optionally, re-raise the exception for further handling
