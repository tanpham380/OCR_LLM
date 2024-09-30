import cv2
import torch
from PIL import Image
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from app_utils.ocr_package.schema import TextDetectionResult, PolygonBox
import numpy as np
from tqdm import tqdm
from typing import List
import io
import time

from controller.llm_vison_future import VinternOCRModel

def crop_text_regions(image: np.ndarray, detection_boxes: List[List[List[float]]]) -> List[Image.Image]:
    """
    Crop the text regions from the image based on the detection bounding boxes.
    
    Parameters:
        image (np.ndarray): The original image in numpy array format.
        detection_boxes (List[List[List[float]]]): List of bounding box coordinates.
    
    Returns:
        List[Image.Image]: List of cropped text regions as PIL Image.
    """
    cropped_images = []
    for box in detection_boxes:
        # Convert float coordinates to integer
        box = np.array(box).astype(int)

        # Extract the bounding box coordinates
        x_min = min(box[:, 0])
        x_max = max(box[:, 0])
        y_min = min(box[:, 1])
        y_max = max(box[:, 1])

        # Crop the image using the bounding box
        cropped_region = image[y_min:y_max, x_min:x_max]
        cropped_pil_image = Image.fromarray(cropped_region)

        cropped_images.append(cropped_pil_image)
    
    return cropped_images

def pil_image_to_byte_io(pil_image: Image.Image) -> io.BytesIO:
    """
    Convert a PIL Image to a file-like object (BytesIO).
    
    Parameters:
        pil_image (Image.Image): The PIL Image to convert.
    
    Returns:
        io.BytesIO: A file-like object that can be passed to functions expecting a file.
    """
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format='JPEG')  # Save the PIL image as JPEG to the byte array
    byte_io.seek(0)  # Reset the pointer to the beginning of the BytesIO object
    return byte_io

def concatenate_images(images: List[Image.Image], direction='vertical') -> Image.Image:
    """
    Concatenate a list of images into one single image.
    
    Parameters:
        images (List[Image.Image]): List of images to concatenate.
        direction (str): 'vertical' or 'horizontal'. Determines the concatenation direction.
    
    Returns:
        Image.Image: Concatenated image.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction == 'horizontal':
        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_image.paste(im, (x_offset, 0))
            x_offset += im.width
    else:
        max_width = max(widths)
        total_height = sum(heights)
        new_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_image.paste(im, (0, y_offset))
            y_offset += im.height
    
    return new_image

# Step 1: Load your image
image_path = "/home/gitlab/ocr/2024_01_22_10_28_55_resize.jpg"
image = cv2.imread(image_path)

# Step 2: Initialize the RapidOCR detector
rapidocr_detector = TextDect_withRapidocr(text_score=0.4, det_use_cuda=False)

# Step 3: Detect text regions
detection_results = rapidocr_detector.detect(image)

# Step 4: Crop the text regions from the image
cropped_images = crop_text_regions(image, detection_results)

# Step 5: Concatenate cropped images into one large image
concatenated_image = concatenate_images(cropped_images, direction='vertical')

# Optional: Save the concatenated image for verification
concatenated_image.save("concatenated_image.jpg")

# Step 6: Initialize the VinternOCRModel
vintern_ocr = VinternOCRModel(model_path="./Vintern-1B-v2")

# Step 7: Process the concatenated image and get the text
prompt = "Trích thông tin từ ảnh, không giải thích"
start_time = time.time()  # Start timing

# Convert the concatenated image to a file-like object (BytesIO)
image_io = pil_image_to_byte_io(concatenated_image)

# Process the image with VinternOCRModel
response = vintern_ocr.process_image(image_io, prompt)

end_time = time.time()  # End timing
processing_time = end_time - start_time

# Print the text response and processing time
print(f"Text from concatenated image: {response}")
print(f"Processing time: {processing_time:.4f} seconds")
