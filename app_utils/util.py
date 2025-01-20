import datetime
from typing import Any, Dict, Tuple
import numpy as np
import cv2
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

def calculate_sex_from_id(id_number):
    # Kiểm tra nếu id_number có đủ độ dài để xác định giới tính
    if len(id_number) >= 4:
        # Lấy số thứ 4 từ mã ID
        gender_digit = int(id_number[3])
        # Xác định giới tính dựa trên số thứ 4
        if gender_digit % 2 == 0:
            return "Nam"
        else:
            return "Nữ"
    # Trường hợp không thể xác định giới tính
    return "Nam"

   
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
                
CHAR_MAPPING = {
    # Existing mappings
    '廕': 'ạ',
    '璽': 'â',
    '《': 'm',
    'ﾃ｢': 'â',
    'ﾃ': 'ă', 
    'ﾆｰ': 'ư',
    '盻': 'ờ',
    '拵': 'ờ',
    'ﾄ': 'Đ',
    '雪': 'ạ',
    'ｺ｡': 'ạ',
    '盻�': 'ọ',
    '皇': 'c',
    # Add common Vietnamese character fixes
    '?' : 'ỏ',
    'ﾄ' : 'Đ',
    'ﾃ³': 'ó',
    'ﾃ¢': 'â',
    'ﾃ½': 'ý'
}

def fix_decoded_text(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Fix Vietnamese text encoding with logging
    Returns: (fixed_text, replacements_made)
    """
    if not text:
        return text, {}
        
    replacements = {}
    fixed = text
    
    for old_char, new_char in CHAR_MAPPING.items():
        if old_char in fixed:
            before = fixed
            fixed = fixed.replace(old_char, new_char)
            replacements[old_char] = new_char
            
    return fixed.strip(), replacements

def fix_name(text: str) -> str:
    """Fix Vietnamese name encoding with logging"""
    parts = text.split('|')
    if len(parts) < 3:
        return text
        
    # Fix all parts that may contain Vietnamese
    for i, part in enumerate(parts):
        fixed, replacements = fix_decoded_text(part)
        parts[i] = fixed
    
    return '|'.join(parts)

def get_card_info(is_front: bool) -> dict:
    """Get card type and issuer based on QR code location"""
    if is_front:
        return {
            "place_of_issue": "Cục trưởng Cục Cảnh sát Quản lý Hành chính về Trật tự Xã hội",
            "type_card": "Căn Cước Công Dân"
        }
    return {
        "place_of_issue": "Bộ Công An",
        "type_card": "Căn Cước"
    }

async def extract_qr_data(front_result: dict, back_result: dict) -> dict:
    """Extract and validate QR data from card images"""
    qr_data = front_result.get("qr_code_text")
    is_front = True
    
    if isinstance(qr_data, list):
        qr_data = qr_data[0] if qr_data else None
    
    # Check if front QR is not detected
    if qr_data == "not_detect" or not qr_data or (isinstance(qr_data, str) and qr_data.isspace()):
        qr_data = back_result.get("qr_code_text")
        if isinstance(qr_data, list):
            qr_data = qr_data[0] if qr_data else None
        
        # If back QR is detected, set is_front to False
        if qr_data and qr_data != "not_detect":
            is_front = False
        # Otherwise keep is_front as True (default case)
            
    if qr_data and qr_data != "not_detect":
        qr_data = fix_name(qr_data)
        
    normalized_qr = normalize_qr_data(qr_data) if qr_data and qr_data != "not_detect" else None
    card_info = get_card_info(is_front)
    
    return {
        "qr_data": normalized_qr,
        **card_info
    }



def normalize_qr_data(qr_data: Any) -> str:
    """Normalize QR code data to string format."""
    if isinstance(qr_data, (list, tuple)):
        return qr_data[0] if qr_data else ""
    return str(qr_data).strip()

def get_ocr_result( ocr_results: Dict[str, Any], key: str, subkey: str) -> str:
    """Safely extract OCR results."""
    return ocr_results.get(key, {}).get(subkey, "")
    
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