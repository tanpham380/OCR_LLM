import asyncio
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
    

async def scan(image_path: List[str]) -> str:
    """
    Scans the ID card (CCCD) image using the EasyOCR controller, processes the text with LLM, 
    and returns the corrected information in JSON format.
    
    Args:
        image_path (List[str]): List containing paths to the two image files (front and back).
    
    Returns:
        str: The formatted JSON string containing corrected OCR data.
    
    Raises:
        Exception: If an error occurs during scanning or processing.
    """
    try:
        start_time = time.time()
        # Process both images concurrently using asyncio.gather
        ocr_result_1, ocr_result_2 = await asyncio.gather(
            process_image(image_path[0]),  # Process front image
            process_image(image_path[1])   # Process back image
        )
        if ocr_result_1["mat_truoc"]:
            combined_ocr_data = {
                "front_side_ocr": ocr_result_1["ocr_text"],
                "front_side_qr": ocr_result_1["qr_code_text"],
                "back_side_ocr": ocr_result_2["ocr_text"],
                "back_side_qr": ocr_result_2["qr_code_text"]
            }
        else:
            combined_ocr_data = {
                "front_side_ocr": ocr_result_2["ocr_text"],
                "front_side_qr": ocr_result_2["qr_code_text"],
                "back_side_ocr": ocr_result_1["ocr_text"],
                "back_side_qr": ocr_result_1["qr_code_text"]
            }
        print(combined_ocr_data)
        llm_controller.set_user_context(combined_ocr_data)  
        llm_controller.set_model('qwen2.5')

        llm_response = await llm_controller.send_message()
        message_content = clean_message_content(llm_response.get('message', {}).get('content', ''))
        print(message_content)
        try:
            message_json = json.loads(message_content)
        except json.JSONDecodeError:
            message_json = {"error": "Failed to parse JSON content"}

        end_time = time.time()
        processing_time = end_time - start_time

        # Add processing time to the response
        llm_response_with_time = {
            "llm_response": message_json,
            "processing_time_seconds": processing_time
        }

        # Clear GPU cache if necessary
        torch.cuda.empty_cache()  # Only if you are using GPU with PyTorch

        return llm_response_with_time

    except Exception as e:
        logger.error(f"An error occurred during the scanning process: {e}")
        raise e


def clean_message_content(message_content: str) -> str:
    """
    Cleans the JSON string by removing code block markers (```json and ```).
    
    Args:
        message_content (str): The raw message content to clean.
    
    Returns:
        str: Cleaned message content without code block markers.
    """
    # Remove code block markers using regular expressions
    return re.sub(r'^```json|```$', '', message_content).strip()

async def process_image(image_path: str) -> dict:
    """
    Processes a single ID card image for OCR, QR code detection, and orientation correction.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        dict: A dictionary containing the OCR, QR code data, and whether it's the front side (mat_truoc).
    """
    try:
        # Load image in an asynchronous fashion if possible
        image, mat_truoc = await asyncio.to_thread(detector_controller.detect, image_path)
        if image is None:
            raise Exception(f"Failed to detect image at {image_path}.")

        # Check if the image is upside down and rotate if necessary
        if mat_truoc:
            up_down_check = await asyncio.to_thread(detector_controller.detect_face_orientation, image)
        else:
            up_down_check = await asyncio.to_thread(detector_controller.detect_card_orientation, image)

        if up_down_check:
            image = await asyncio.to_thread(rotate_image, image, 180)

        # Parallelize QR code and OCR scanning
        qr_code_text, ocr_text = await asyncio.gather(
            asyncio.to_thread(detector_controller.read_QRcode, image),
            asyncio.to_thread(detector_controller.get_ocr().scan_image, image, ["package_ocr"])
        )

        # Return OCR data, QR data, and whether it's the front side (mat_truoc)
        return {
            "ocr_text":await ocr_text,
            "qr_code_text": await qr_code_text if qr_code_text else " ",
            "mat_truoc": mat_truoc
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise e
