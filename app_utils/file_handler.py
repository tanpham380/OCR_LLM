import datetime
import io
import os
import shutil
import uuid
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Any, Optional, Tuple, Union
import requests
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from app_utils.logging import get_logger
from config import ALLOWED_EXTENSIONS, SAVE_IMAGES, TEMP_DIR

logger = get_logger(__name__)
# Initialize the logger

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

def generate_filename(extension: str = 'jpg') -> str:
    """
    Generates a filename using the format: day_month_year_uidv4.

    Args:
        extension (str): The file extension (e.g., 'jpg', 'png').

    Returns:
        str: The generated filename.
    """
    current_time = datetime.datetime.now().strftime("%d_%m_%Y")  # Format: day_month_year
    random_string = uuid.uuid4().hex[:8]  # Generate a random string using UUID4
    return f"{current_time}_{random_string}.{extension}"

def clear_folder_content(foldername: str) -> None:
    """
    Clears the contents of the specified folder.

    Args:
        foldername (str): The path to the folder to be cleared.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        PermissionError: If there's no permission to delete the folder contents.
    """
    try:
        if os.path.exists(foldername):
            shutil.rmtree(foldername)
            os.makedirs(foldername)  # Recreate the empty folder
        else:
            raise FileNotFoundError(f"Folder not found: {foldername}")
    except PermissionError as pe:
        raise pe
    except Exception as e:
        raise ImageProcessingError(f"Failed to clear folder: {str(e)}")

def allowed_file(filename: str) -> bool:
    """
    Returns:
        bool: True if the file is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_file(file: FileStorage) -> str:
    """
    Saves an allowed file to a temporary directory.

    Args:
        file (FileStorage): The file to be saved.

    Returns:
        str: The path to the saved file.

    Raises:
        ValueError: If the file type is not allowed.
        ImageProcessingError: If there's an error processing the image.
    """
    if not allowed_file(file.filename):
        raise ValueError("File type not allowed")

    filename = secure_filename(file.filename)

    try:
        file.seek(0)  # Ensure the file pointer is at the start
        np_image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img is None:
            file.seek(0)  # Reset file pointer to start
            img = Image.open(file)
            img.verify()  # Verify that the image is not corrupted
            img = img.convert("RGB")  # Convert to RGB to handle different formats
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        filepath = save_image(img, TEMP_DIR)
        
        return filepath

    except cv2.error as cv_error:
        raise ImageProcessingError(f"OpenCV error while processing the file: {cv_error}")
    except Exception as e:
        raise ImageProcessingError(f"Failed to save file: {str(e)}")

def save_image(image: Union[str, np.ndarray, Image.Image], save_dir: str = TEMP_DIR, format: str = 'PNG', print_path: bool = True) -> str:
    """
    Saves an image to the specified directory with a unique filename format (day_month_year_uidv4).

    Args:
        image (Union[str, np.ndarray, Image.Image]): The image to be saved.
        save_dir (str): The directory where the image will be saved.
        format (str): The format to save the image in (e.g., 'JPEG', 'PNG'). Default is 'PNG'.
        print_path (bool): Whether to print the saved image path. Default is True.

    Returns:
        str: The full path to the saved image file.

    Raises:
        ValueError: If the save directory is invalid or image format is unsupported.
        ImageProcessingError: If there's an error saving the image.
    """
    if not os.path.isdir(save_dir):
        raise ValueError(f"Save directory is not a valid directory: {save_dir}")
    
    extension = format.lower() if format.lower() in ['jpeg', 'jpg', 'png'] else "png"
    save_path = os.path.join(save_dir, generate_filename(extension))
    
    try:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ImageProcessingError(f"Failed to read image from the path: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            Image.fromarray(img).save(save_path, format=format)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 3:  
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4: 
                    img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    img = image  
            else:
                raise ValueError("Unsupported number of channels in NumPy array.")
            Image.fromarray(img).save(save_path, format=format)
        elif isinstance(image, Image.Image):
            image.convert("RGB").save(save_path, format=format)
        else:
            raise ValueError("Unsupported image format. Please provide a file path, NumPy array, or PIL Image.")
        file_size = os.path.getsize(save_path)
        file_permissions = oct(os.stat(save_path).st_mode)[-3:]
        
        
        if print_path:
            logger.info(f"Saved image details - Size: {file_size} bytes, Permissions: {file_permissions}")
            logger.info(f"Image saved successfully at {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        raise ImageProcessingError(f"Failed to save image: {str(e)}")    
    
    

def convert_image(image_data: Union[str, np.ndarray, Image.Image]) -> bytes:
    """
    Converts an image from a file path, NumPy array, or Pillow image to bytes.

    Args:
        image_data: The image data to be converted.

    Returns:
        bytes: The image in PNG format.

    Raises:
        ImageProcessingError: If there's an error converting the image.
    """
    try:
        _, image = load_and_preprocess_image(image_data)
        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()
    except Exception as e:
        raise ImageProcessingError(f"Error converting image: {e}")
    
    
class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

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
                raise ImageProcessingError("Failed to decode image from bytes.")
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
                raise ImageProcessingError("Unable to fix the image loading issue.")
            return img
        except Exception as e:
            raise ImageProcessingError(f"Failed to load and fix image: {str(e)}")


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
    """
    Loads an image and applies preprocessing while keeping the original unchanged.

    Args:
        image_input (Any): The image data. Can be a file path (str), a NumPy array,
                           a Pillow Image, or a bytes object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - processed_img (np.ndarray): The preprocessed image.
            - original_img (np.ndarray): The original image.
    """
    try:
        original_img = load_image(image_input)
        processed_img = preprocess_image(original_img.copy())  
        return processed_img, original_img
    except Exception as e:
        raise ImageProcessingError(f"Error loading and preprocessing image: {e}")
    
def crop_image_qr(image: np.ndarray, detection: dict, margin: int = 20) -> np.ndarray:
    """
    Tries to crop the QR code area from the image using polygon points first. 
    If it fails, falls back to using bounding box coordinates.

    Args:
        image (np.ndarray): Input image containing the QR code.
        detection (dict): Detection dictionary containing the bounding box and polygon information.
        margin (int): Margin to add around the detected QR code area.

    Returns:
        np.ndarray: Cropped image with the QR code.
    """
    # Attempt to use polygon points for cropping
    if 'polygon_xy' in detection and len(detection['polygon_xy']) >= 4:
        try:
            return crop_image_using_polygon(image, detection['polygon_xy'], margin)
        except Exception as e:
            return crop_image_with_margin(image, detection['bbox_xyxy'], margin)

    # Fallback to bounding box cropping if no polygon is available or an error occurred
    return crop_image_with_margin(image, detection['bbox_xyxy'], margin)

def crop_image_with_margin(image: np.ndarray, bbox_xyxy: np.ndarray, margin: int = 20) -> np.ndarray:
    """
    Crops the QR code area from the image with a margin around the bounding box.

    Args:
        image (np.ndarray): Input image containing the QR code.
        bbox_xyxy (np.ndarray): Bounding box coordinates [x_min, y_min, x_max, y_max].
        margin (int): Margin to add around the detected QR code bounding box.

    Returns:
        np.ndarray: Cropped image with margin around the QR code.
    """
    # Extract bounding box coordinates and add margin
    x_min, y_min, x_max, y_max = bbox_xyxy.astype(int)
    
    # Add margin and ensure we don't go out of image bounds
    height, width = image.shape[:2]
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, width)
    y_max = min(y_max + margin, height)
    
    # Crop the image with the expanded bounding box
    return image[y_min:y_max, x_min:x_max]
def crop_image_using_polygon(image: np.ndarray, detection_info: dict, margin: int = 20) -> np.ndarray:
    """
    Crops the QR code area from the image using polygon points and applies a perspective transform.

    Args:
        image (np.ndarray): Input image containing the QR code.
        detection_info (dict): Detection information dictionary from QReader.detect() method.
        margin (int): Margin to add around the detected QR code polygon.

    Returns:
        np.ndarray: Cropped and transformed image with the QR code.
    """
    # Extract the padded_quad_xy from detection_info
    quad_xy = detection_info['padded_quad_xy'].astype(np.float32)

    # Calculate the bounding box from the polygon and add margin
    x_min = max(int(np.min(quad_xy[:, 0])) - margin, 0)
    y_min = max(int(np.min(quad_xy[:, 1])) - margin, 0)
    x_max = min(int(np.max(quad_xy[:, 0])) + margin, image.shape[1])
    y_max = min(int(np.max(quad_xy[:, 1])) + margin, image.shape[0])
    
    # Get the cropped region containing the QR code
    cropped_qr = image[y_min:y_max, x_min:x_max]

    # Adjust quad_xy points relative to the cropped region
    adjusted_pts = quad_xy - np.array([x_min, y_min], dtype="float32")
    
    # Define the destination points for the perspective transform
    rect = np.array([[0, 0], 
                     [cropped_qr.shape[1] - 1, 0], 
                     [cropped_qr.shape[1] - 1, cropped_qr.shape[0] - 1], 
                     [0, cropped_qr.shape[0] - 1]], dtype="float32")
    
    # Apply perspective transform to straighten the QR code
    perspective_transform = cv2.getPerspectiveTransform(adjusted_pts, rect)
    warped_qr = cv2.warpPerspective(image, perspective_transform, (cropped_qr.shape[1], cropped_qr.shape[0]))

    return warped_qr


def process_qr_image(img: np.ndarray, target_size: int = 478) -> np.ndarray:
    """
    Resize the input image to fit within a square background of specified size while keeping the aspect ratio.
    Adds a white margin around the image.

    Args:
        img (np.ndarray): Input image as a numpy array.
        target_size (int): Desired size of the output image (both width and height). Default is 478 pixels.

    Returns:
        np.ndarray: Processed image with a white background as a numpy array.
    """
    h, w = img.shape[:2]
    aspect_ratio = min(target_size / w, target_size / h)
    new_w, new_h = int(w * aspect_ratio), int(h * aspect_ratio)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if len(resized_img.shape) == 3:  
        background_img = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    else: 
        background_img = np.full((target_size, target_size), 255, dtype=np.uint8)

    # Step 3: Center the resized image on the white background
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    background_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img

    return background_img


# def process_qr_image(img: np.ndarray, target_size: int = 478) -> np.ndarray:
#     """
#     Adds the input image to the center of a white background with the specified size.

#     Args:
#         img (np.ndarray): Input image as a numpy array.
#         target_size (int): Desired size of the output image (both width and height). Default is 478 pixels.

#     Returns:
#         np.ndarray: Processed image with the original image centered on a white background.
#     """
#     if len(img.shape) == 3:  # If the image has 3 channels (color)
#         background_img = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
#     else:  # If the image is grayscale
#         background_img = np.full((target_size, target_size), 255, dtype=np.uint8)

#     h, w = img.shape[:2]

#     start_x = (target_size - w) // 2
#     start_y = (target_size - h) // 2

#     background_img[start_y:start_y + h, start_x:start_x + w] = img

#     return background_img


def convert_to_silver(img: np.ndarray) -> np.ndarray:
    """
    Converts an input image to a silver (light gray) color scheme.

    Args:
        img (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Image converted to a silver color scheme.
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale image to the range [0, 255]
    normalized_gray = cv2.normalize(gray_img, None, 200, 255, cv2.NORM_MINMAX)

    # Convert normalized grayscale back to BGR format for display
    silver_img = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)

    return silver_img