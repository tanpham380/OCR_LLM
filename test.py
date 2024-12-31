import json
import cv2
import gradio as gr
import time
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from controller.llm_vison_future import VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import torch
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from typing import List
import numpy as np
from PIL import Image
from app_utils.ocr_package.schema import TextDetectionResult, PolygonBox
from controller.qwen_vison import EraXVLOcrModel
from prompt import CCCD_FRONT_PROMPT
# Initialize the LLM Vision model

llm_vison = EraXVLOcrModel(
    # model_path="app_utils/weights/Vintern-3B-beta",
    device=torch.device("cuda:0")  # Use specific GPU
)

# llm_vison = VinternOCRModel("app_utils/weights/Vintern-3B-beta")
idcard_detect = ImageRectify(crop_expansion_factor=0.0)
orientation_engine = RapidOrientation()
det_processor = TextDect_withRapidocr(text_score = 0.6 , det_use_cuda = True)


def adjust_bbox_height(bbox_coords, height_adjustment_factor=0.5):
    # Get the y-coordinates from the bbox coordinates
    y_coords = [y for x, y in bbox_coords]
    min_y, max_y = min(y_coords), max(y_coords)
    height = max_y - min_y
    # Expand the height by the adjustment factor
    adjusted_min_y = min_y - (height * height_adjustment_factor / 2)
    adjusted_max_y = max_y + (height * height_adjustment_factor / 2)
    
    # Adjust the bbox with the new y-coordinates
    adjusted_bbox = []
    for (x, y) in bbox_coords:
        if y == min_y:
            adjusted_bbox.append([x, adjusted_min_y])
        elif y == max_y:
            adjusted_bbox.append([x, adjusted_max_y])
        else:
            adjusted_bbox.append([x, y])  # No change for other points
    return adjusted_bbox

def batch_text_detection(images: List[Image.Image]) -> List[TextDetectionResult]:
    results = []
    for img in images:
        # Convert PIL image to numpy array
        img_array = np.array(img)
        #  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect text using RapidOCR
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        elif img_array.ndim != 3:
            raise ValueError("Invalid image dimensions")
            
        det_results = det_processor.detect(img_array)
        
        bboxes = []
        for bbox_coords in det_results:
            try:
                adjusted_bbox_coords = adjust_bbox_height(bbox_coords, height_adjustment_factor=0.25)
                polygon_box = PolygonBox(polygon=adjusted_bbox_coords)
                bboxes.append(polygon_box)
            except ValueError as e:
                continue  # Skip invalid bbox
        
        result = TextDetectionResult(
            bboxes=bboxes,
            vertical_lines=[],  # Add logic for detecting vertical lines if needed
            heatmap=None,       # Create a heatmap if needed
            affinity_map=None,  # Create an affinity map if needed
            image_bbox=[0, 0, img.width, img.height]
        )
        results.append(result)
    return results
def format_vision_output(text_from_vision_model: str) -> str:
    """Format vision model output as pretty JSON"""
    try:
        # Parse JSON string
        json_data = json.loads(text_from_vision_model)
        
        # Format with indentation for readability
        formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
        return formatted_json
    except json.JSONDecodeError:
        # Return original text if not valid JSON
        return text_from_vision_model

def process_image(image):
    start_time = time.time()
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Process image detection and orientation
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_image, is_front = idcard_detect.detect(image_bgr)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process orientation
    orientation_res, _ = orientation_engine(detected_image)
    orientation_res = float(orientation_res)
    
    if orientation_res != 0:
        detected_image = rotate_image(detected_image, orientation_res)
    
    # Crop if not front side
    if not is_front:
        height = detected_image.shape[0] // 2
        detected_image = detected_image[:height, :]
    
    # Convert to RGB for vision model
    image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    
    # Process with vision model
    text_from_vision_model = llm_vison.process_image(image_rgb ,CCCD_FRONT_PROMPT )
    formatted_text = format_vision_output(text_from_vision_model)

    # Convert back for display
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    
    total_time = time.time() - start_time
    processing_time_message = f"Thời gian xử lý: {total_time:.2f} giây"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return detected_image_rgb, formatted_text, processing_time_message

# Tạo giao diện Gradio
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Tải lên một ảnh"),
    outputs=[
        gr.Image(type="numpy", label="Ảnh sau khi xử lý"),
        gr.Textbox(label="Văn bản từ mô hình Vision"),
        gr.Textbox(label="Thời gian xử lý")
    ],
    title="Trò chuyện dựa trên Vision với LLM",
    description="Tải lên một ảnh và tương tác với LLM dựa trên Vision để tạo văn bản và xem thời gian xử lý."
)

# Khởi chạy ứng dụng
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
