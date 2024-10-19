import datetime
import io
import os
import shutil
import uuid
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import requests
from werkzeug.datastructures import FileStorage

from app_utils.logging import get_logger
from config import ALLOWED_EXTENSIONS, SAVE_IMAGES, TEMP_DIR

logger = get_logger(__name__)

def generate_filename(extension: str = 'jpg') -> str:
    """Generates a filename using the format: day_month_year_uidv4."""
    return f"{datetime.datetime.now().strftime('%d_%m_%Y')}_{uuid.uuid4().hex[:8]}.{extension}"

def clear_folder_content(foldername: str) -> None:
    """Clears the contents of the specified folder."""
    try:
        if os.path.exists(foldername):
            shutil.rmtree(foldername)
            os.makedirs(foldername)
        else:
            raise FileNotFoundError(f"Folder not found: {foldername}")
    except Exception as e:
        raise Exception(f"Failed to clear folder: {str(e)}")

def allowed_file(filename: str) -> bool:
    """Checks if the file is allowed based on its extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_file(file: FileStorage) -> str:
    """Saves an allowed file to a temporary directory."""
    if not allowed_file(file.filename):
        raise ValueError("File type not allowed")

    try:
        file.seek(0)
        np_image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img is None:
            file.seek(0)
            img = Image.open(file).convert("RGB")
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return save_image(img, TEMP_DIR)
    except Exception as e:
        raise Exception(f"Failed to save file: {str(e)}")

# def save_image(image: Union[str, np.ndarray, Image.Image], save_dir: str = TEMP_DIR, format: str = 'PNG', print_path: bool = True) -> str:
#     """Saves an image to the specified directory with a unique filename."""
#     if not os.path.isdir(save_dir):
#         raise ValueError(f"Invalid save directory: {save_dir}")
    
#     extension = format.lower() if format.lower() in ['jpeg', 'jpg', 'png'] else "png"
#     save_path = os.path.join(save_dir, generate_filename(extension))
    
#     try:
#         if isinstance(image, str):
#             img = cv2.imread(image)
#             if img is None:
#                 raise Exception(f"Failed to read image from path: {image}")
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             Image.fromarray(img).save(save_path, format=format)
#         elif isinstance(image, np.ndarray):
#             img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
#             Image.fromarray(img).save(save_path, format=format)
#         elif isinstance(image, Image.Image):
#             image.convert("RGB").save(save_path, format=format)
#         else:
#             raise ValueError("Unsupported image format.")
        
#         if print_path:
#             file_size = os.path.getsize(save_path)
#             file_permissions = oct(os.stat(save_path).st_mode)[-3:]
#             logger.info(f"Saved image - Size: {file_size} bytes, Permissions: {file_permissions}")
#             logger.info(f"Image saved at {save_path}")
#         return save_path
#     except Exception as e:
#         logger.error(f"Failed to save image: {str(e)}")
#         raise Exception(f"Failed to save image: {str(e)}")

def save_image(image: Union[str, np.ndarray, Image.Image], save_dir: str = TEMP_DIR, format: str = 'PNG', print_path: bool = True) -> str:
    """Saves an image to the specified directory with a unique filename."""
    if not os.path.isdir(save_dir):
        raise ValueError(f"Invalid save directory: {save_dir}")
    
    extension = format.lower() if format.lower() in ['jpeg', 'jpg', 'png'] else "png"
    save_path = os.path.join(save_dir, generate_filename(extension))
    
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise Exception(f"Failed to read image from path: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img).save(save_path, format=format)
        elif isinstance(image, np.ndarray):
            # Check if the image is grayscale or RGB
            if len(image.shape) == 2:  # Grayscale
                img = image
            elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Unsupported image format.")
            
            Image.fromarray(img).save(save_path, format=format)
        elif isinstance(image, Image.Image):
            image.convert("RGB").save(save_path, format=format)
        else:
            raise ValueError("Unsupported image format.")
        
        if print_path:
            file_size = os.path.getsize(save_path)
            file_permissions = oct(os.stat(save_path).st_mode)[-3:]
            logger.info(f"Saved image - Size: {file_size} bytes, Permissions: {file_permissions}")
            logger.info(f"Image saved at {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        raise Exception(f"Failed to save image: {str(e)}")


def load_image(image_input: Any) -> np.ndarray:
    """
    Loads an image from various sources and tries to handle errors gracefully.

    Supports loading images from:
        - File paths (local or URLs starting with 'http://' or 'https://')
        - NumPy arrays (representing images)
        - Pillow Image objects
        - Byte strings (representing image data)

    Args:
        image_input (Any): The image data. Can be a file path (str), a NumPy array,
                           a Pillow Image, or a bytes object.

    Returns:
        np.ndarray: The loaded image as a NumPy array in RGB format.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ImageProcessingError: If there's an error loading or converting the image.
        ValueError: If the input type is not supported or if the image data is invalid.
    """
    

    def attempt_color_fix(img):
        """Attempts to fix color issues if the initial load fails."""
        if img.ndim == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    try:
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            elif not os.path.exists(image_input):
                raise FileNotFoundError(f"File not found: {image_input}")
            else:
                img = Image.open(image_input)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise Exception("Failed to decode image from bytes.")
        else:
            raise ValueError("Unsupported image input type.")
        if img is None or len(img.shape) < 2 or len(img.shape) > 3:
            raise ValueError("Invalid image format. The image could not be loaded correctly.")
        return img 

    except Exception as e:
        try:
            if isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Force color load
                img = attempt_color_fix(img)  # Apply color fix
            elif isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")  # Convert to RGB to fix color issues
                img = np.array(img)
                
                
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input.convert("RGB"))  # Convert to RGB
            elif isinstance(image_input, np.ndarray):
                img = attempt_color_fix(image_input)  # Apply color fix for NumPy array
            else:
                raise Exception("Unable to fix the image loading issue.")
            return img
        except Exception as e:
            raise Exception(f"Failed to load and fix image: {str(e)}")

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing to an image, enhancing contrast and sharpness.

    Args:
        img (np.ndarray): The image as a NumPy array in RGB format.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array.
    """
    # Convert to PIL Image for processing
    pil_img = Image.fromarray(img)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.5)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
    
    # Convert back to NumPy array
    processed_img = np.array(pil_img)
    return processed_img

def load_and_preprocess_image(image_input: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Loads an image and applies preprocessing while keeping the original unchanged."""
    try:
        original_img = load_image(image_input)
        processed_img = preprocess_image(original_img.copy())
        return processed_img, original_img
    except Exception as e:
        raise Exception(f"Error loading and preprocessing image: {e}")

def crop_image_qr(image: np.ndarray, detection: dict, margin: int = 0) -> np.ndarray:
    """Crops the QR code area from the image."""
    if 'polygon_xy' in detection and len(detection['polygon_xy']) >= 4:
        try:
            return crop_image_using_polygon(image, detection['polygon_xy'], margin)
        except Exception:
            pass
    return crop_image_with_margin(image, detection['bbox_xyxy'], margin)

def crop_image_with_margin(image: np.ndarray, bbox_xyxy: np.ndarray, margin: int = 20) -> np.ndarray:
    """Crops the QR code area from the image with a margin around the bounding box."""
    x_min, y_min, x_max, y_max = bbox_xyxy.astype(int)
    height, width = image.shape[:2]
    x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
    x_max, y_max = min(x_max + margin, width), min(y_max + margin, height)
    return image[y_min:y_max, x_min:x_max]

def crop_image_using_polygon(image: np.ndarray, detection_info: dict, margin: int = 20) -> np.ndarray:
    """Crops the QR code area from the image using polygon points and applies a perspective transform."""
    quad_xy = detection_info['padded_quad_xy'].astype(np.float32)
    x_min, y_min = np.min(quad_xy, axis=0).astype(int) - margin
    x_max, y_max = np.max(quad_xy, axis=0).astype(int) + margin
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, image.shape[1]), min(y_max, image.shape[0])
    
    cropped_qr = image[y_min:y_max, x_min:x_max]
    adjusted_pts = quad_xy - np.array([x_min, y_min])
    
    rect = np.array([[0, 0], [cropped_qr.shape[1] - 1, 0], 
                     [cropped_qr.shape[1] - 1, cropped_qr.shape[0] - 1], 
                     [0, cropped_qr.shape[0] - 1]], dtype="float32")
    
    perspective_transform = cv2.getPerspectiveTransform(adjusted_pts, rect)
    return cv2.warpPerspective(image, perspective_transform, (cropped_qr.shape[1], cropped_qr.shape[0]))

def process_qr_image(img: np.ndarray) -> np.ndarray:
    """Processes the input image to optimize QR code reading."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.fastNlMeansDenoising(binary_img, None, 10, 7, 21)


def scale_up_img(img: np.ndarray, target_size: int = 1048) -> np.ndarray:
    """Scales up the image to a target size if it's smaller than the specified target size."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    if min(img_pil.size) >= target_size:
        return img
    
    aspect_ratio = min(target_size / dim for dim in img_pil.size)
    new_size = tuple(int(dim * aspect_ratio) for dim in img_pil.size)
    img_resized = img_pil.resize(new_size, Image.Resampling.LANCZOS)
    
    return cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)


def convert_image(image_data: np.ndarray) -> bytes:
    """Converts an np.ndarray image to bytes in PNG format."""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        with io.BytesIO() as output:
            img_pil.save(output, format='PNG')
            return output.getvalue()
    except Exception as e:
        raise Exception(f"Error converting image: {e}")



def crop_text_regions(image: np.ndarray, detection_boxes: list) -> list:
    """Crop the text regions from the image based on the detection bounding boxes."""
    cropped_images = []
    for box in detection_boxes:
        # Swap x and y since box coordinates are in [x, y] format
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Ensure coordinates are within image boundaries
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)

        if y_min < y_max and x_min < x_max:
            cropped_images.append(image[y_min:y_max, x_min:x_max])

    return cropped_images




