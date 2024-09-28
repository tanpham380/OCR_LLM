from collections import Counter
import datetime
import math
import re
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import torch

from datetime import datetime

def calculate_expiration_date(date_of_birth_str):
    date_of_birth = datetime.strptime(date_of_birth_str, "%d/%m/%Y")
    today = datetime.today()
    
    # Calculate age
    age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))

    # Determine the expiration milestone based on age
    if age < 25:
        milestone_age = 25
    elif age < 40:
        milestone_age = 40
    elif age < 60:
        milestone_age = 60
    else:
        # If over 60 years old, the ID card has no expiration date
        return ""

    # Calculate expiration date based on the milestone
    expiration_date = date_of_birth.replace(year=date_of_birth.year + milestone_age)
    expiration_date_str = expiration_date.strftime("%d/%m/%Y")
    
    return expiration_date_str



   
def select_device():
    """
    Automatically selects 'cuda' if a GPU is available; otherwise, it selects 'cpu'.
    
    Returns:
        device (torch.device): The device to use for computations.
    """
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda:0")  # Selects the first GPU
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu") 
    
    


def non_max_suppression_fast(boxes, labels, overlapThresh):
    """
    Performs non-maximum suppression to eliminate redundant overlapping bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes.
        labels (list): List of labels corresponding to the bounding boxes.
        overlapThresh (float): Overlap threshold for suppression.

    Returns:
        tuple: Filtered boxes and corresponding labels.
    """
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), [labels[idx] for idx in pick]



def processImageBeforeRecognitionText(image_path):
    """
    Processes an image before text recognition by applying various preprocessing techniques.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image: Processed image.
    """
    im = Image.open(image_path)
    im = im.resize((im.width * 8, im.height * 8), Image.BICUBIC)  # Bicubic interpolation
    im = ImageOps.equalize(im)  # Histogram equalization
    im = im.filter(ImageFilter.GaussianBlur(1))  # Remove noise using gaussian filter
    im = ImageOps.grayscale(im)  # Convert to grayscale
    return im



def fix_date_format(date_str):
    # Try to match the date format "ddmmyyyy"
    match = re.match(r"(\d{1,2})(\d{1,2})(\d{4})", date_str)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    
    # Try to match the date format "dd/mm/yyyy"
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{1,4})", date_str)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    
    # Try to match the date format "ddmmyy"
    match = re.match(r"(\d{1,2})(\d{1,2})(\d{2})", date_str)
    if match:
        day, month, year = match.groups()
        year = f"20{year}" if int(year) < 50 else f"19{year}"
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    
    # Try to match the date format "ddmmyy"
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{1,4})", date_str)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    match = re.match(r"(\d{1,2})/(\d{1,2})(\d{4})", date_str)
    if match:
        day, month, year = match.groups()
        if len(year) == 4:
            return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        else:
            return f"{day.zfill(2)}/{month.zfill(2)}/{year.zfill(4)}"
    
    return date_str 

def is_mostly_upside_down( ocr_text: str) -> bool:
    """
    Checks if the majority of detected text is upside down using a heuristic approach based on character inversion, irregular spacing, and language features.
    Args:
        ocr_text (str): OCR text extracted from the image.
    Returns:
        bool: True if the text is mostly upside down, False otherwise.
    """
    inverted_char_map = {
        'n': 'u', 'u': 'n', 'p': 'd', 'd': 'p', 'b': 'q', 'q': 'b',
        'm': 'w', 'w': 'm', 'v': '^', '^': 'v', 'a': 'ɒ', 'ɒ': 'a',
        'e': 'ǝ', 'ǝ': 'e', 's': 'z', 'z': 's', '6': '9', '9': '6'
    }
    inverted_count = sum(
        1 for char in ocr_text if char in inverted_char_map and ocr_text.count(inverted_char_map[char]) > 0
    )
    irregular_spacing_count = sum(1 for i in range(1, len(ocr_text)) if ocr_text[i-1].isalnum() and ocr_text[i] in ",.;:")
    common_patterns = ['the', 'and', 'ing', 'ion', 'tion', 'at', 'to', 'is']
    pattern_count = sum(ocr_text.count(pattern) for pattern in common_patterns)

    inverted_threshold = 0.2  # 20% of characters are inverted
    irregular_threshold = 0.1  # 10% of spacing is irregular
    common_pattern_threshold = 0.05  # Less than 5% matches common patterns
    
    inverted_ratio = inverted_count / len(ocr_text)
    irregular_ratio = irregular_spacing_count / len(ocr_text)
    common_pattern_ratio = pattern_count / len(ocr_text)
    return (
        inverted_ratio > inverted_threshold or
        irregular_ratio > irregular_threshold or
        common_pattern_ratio < common_pattern_threshold
    )
    
    
def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates the image by the given angle.
    """
    height, width = img.shape[:2]
    image_center = (width / 2, height / 2)
    
    # Rotation matrix computation
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    # Determine bounding box after rotation
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix to new bounds
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    
    # Perform the affine transformation (rotation)
    rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
    return rotated_img