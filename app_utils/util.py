import datetime
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