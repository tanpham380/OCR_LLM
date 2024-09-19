import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import re
from typing import List
import numpy as np
import torch
from app_utils.file_handler import save_image
from app_utils.logging import get_logger
from app_utils.util import rotate_image
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
        if mat_sau:
            up_down_check = await detector_controller.detect_card_orientation(image)
        else:
            up_down_check = detector_controller.detect_face_orientation(image)
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
        start_time = time.time()
        db_manager = g.db_manager
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
        if not db_manager:
            raise RuntimeError("Database manager not initialized.")

        ocr_result_id = await db_manager.insert_ocr_result(combined_ocr_data)

        # Convert images to base64 in parallel
        image_base64_tasks = [
            db_manager.image_to_base64(front_result["image_path"]),
            db_manager.image_to_base64(back_result["image_path"])
        ]
        front_image_base64, back_image_base64 = await asyncio.gather(*image_base64_tasks)

        # Insert images into the database
        await asyncio.gather(
            db_manager.insert_image(ocr_result_id, 'front', front_image_base64),
            db_manager.insert_image(ocr_result_id, 'back', back_image_base64)
        )

        # LLM processing
        context = await asyncio.to_thread(llm_controller.set_user_context, combined_ocr_data)
        await db_manager.insert_user_context(ocr_result_id, context)

        llm_controller.set_model('qwen2.5')
        llm_response = await llm_controller.send_message()

        message_content = clean_message_content(llm_response.get('message', {}).get('content', ''))
        message_json = json.loads(message_content)

        processing_time = time.time() - start_time
        llm_response_with_time = {
            "llm_response": message_json,
            "processing_time_seconds": processing_time
        }

        logger.info(f"LLM response with processing time: {llm_response_with_time}")

        # Clear GPU cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await db_manager.insert_scan_result(llm_response_with_time)

        return llm_response_with_time

    except Exception as e:
        logger.error(f"An error occurred during the scanning process: {e}")
        raise

async def process_image(image_path: str) -> dict:
    """
    Processes a single ID card image for OCR, QR code detection, and orientation correction.
    """
    try:
        # Remove asyncio.to_thread for model inference
        image, mat_truoc = detector_controller.detect(image_path)
        if image is None:
            raise ValueError(f"Failed to detect image at {image_path}.")

        # Determine orientation
        if mat_truoc:
            up_down_check = detector_controller.detect_face_orientation(image)
        else:
            up_down_check = detector_controller.detect_card_orientation(image)

        if up_down_check:
            image = rotate_image(image, 180)

        # Parallelize QR code and OCR scanning
        qr_code_text_task = asyncio.to_thread(detector_controller.read_QRcode, image)
        ocr_text_task = asyncio.to_thread(detector_controller.get_ocr().scan_image, image, ["package_ocr"])
        qr_code_text, ocr_text = await asyncio.gather(qr_code_text_task, ocr_text_task)

        return {
            "ocr_text": await ocr_text,
            "qr_code_text": await qr_code_text or " ",
            "mat_truoc": mat_truoc,
            "image_path": image_path  # Include image path for later use
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise


def clean_message_content(message_content: str) -> str:
    """
    Cleans the JSON string by removing code block markers (```json and ```).
    """
    return re.sub(r'^```json|```$', '', message_content).strip()