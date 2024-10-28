import cv2
import gradio as gr
import time
from app_utils.image_rectifier_package.image_rectifier import ImageRectify
from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from controller.llm_vison_future import VinternOCRModel
from app_utils.rapid_orientation_package.rapid_orientation import RapidOrientation
from app_utils.util import rotate_image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List
import numpy as np
from PIL import Image
from app_utils.ocr_package.schema import TextDetectionResult, PolygonBox

# Initialize the process group for DDP
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)

# Initialize the LLM Vision model with DDP
llm_vison = VinternOCRModel("app_utils/weights/Vintern-3B-v1-phase4-ocr", device=device)
llm_vison.model = llm_vison.model.to(device)  # Move the entire model to the device
llm_vison.model = DDP(llm_vison.model, device_ids=[local_rank], output_device=local_rank) 

idcard_detect = ImageRectify(crop_expansion_factor=0.0)
orientation_engine = RapidOrientation()
det_processor = TextDect_withRapidocr(text_score=0.6, det_use_cuda=True)

def adjust_bbox_height(bbox_coords, height_adjustment_factor=0.5):
    """
    Adjusts the height of a bounding box.

    Args:
      bbox_coords: A list of coordinates representing the bounding box.
      height_adjustment_factor: The factor by which to adjust the height.

    Returns:
      A list of coordinates representing the adjusted bounding box.
    """
    y_coords = [y for x, y in bbox_coords]
    min_y, max_y = min(y_coords), max(y_coords)
    height = max_y - min_y
    adjusted_min_y = min_y - (height * height_adjustment_factor / 2)
    adjusted_max_y = max_y + (height * height_adjustment_factor / 2)
    
    adjusted_bbox = []
    for (x, y) in bbox_coords:
        if y == min_y:
            adjusted_bbox.append([x, adjusted_min_y])
        elif y == max_y:
            adjusted_bbox.append([x, adjusted_max_y])
        else:
            adjusted_bbox.append([x, y])
    return adjusted_bbox

def batch_text_detection(images: List[Image.Image]) -> List[TextDetectionResult]:
    """
    Performs text detection on a batch of images.

    Args:
      images: A list of PIL Images.

    Returns:
      A list of TextDetectionResult objects.
    """
    results = []
    for img in images:
        img_array = np.array(img)
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
                continue
        
        result = TextDetectionResult(
            bboxes=bboxes,
            vertical_lines=[],
            heatmap=None,
            affinity_map=None,
            image_bbox=[0, 0, img.width, img.height]
        )
        results.append(result)
    return results

def process_image(image):
    """
    Processes an image with the LLM Vision model.

    Args:
      image: A NumPy array representing the image.

    Returns:
      A tuple containing the processed image, text from the vision model, and processing time.
    """
    start_time = time.time()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_image, is_front = idcard_detect.detect(image_bgr)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orientation_res, _ = orientation_engine(detected_image)
    orientation_res = float(orientation_res)

    if orientation_res != 0:
        detected_image = rotate_image(detected_image, orientation_res)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not is_front:
        height = detected_image.shape[0] // 2
        detected_image = detected_image[:height, :]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    text_from_vision_model = llm_vison.process_image(image_gray)

    detected_image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB)

    total_time = time.time() - start_time
    processing_time_message = f"Thời gian xử lý: {total_time:.2f} giây"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detected_image_rgb, text_from_vision_model, processing_time_message

# Create Gradio interface
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

# Launch the app with torch.distributed.launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")