from app_utils.ocr_package.model.detection.text_detect import TextDect_withRapidocr
from typing import List
import numpy as np
from PIL import Image
from app_utils.ocr_package.schema import TextDetectionResult, PolygonBox

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

def batch_text_detection(images: np.ndarray, rapidocr_detector: TextDect_withRapidocr) -> List[List[List[float]]]:
    """
    Detects text and returns adjusted bounding box coordinates.

    Args:
        images (np.ndarray): Input image in numpy array format.
        rapidocr_detector (TextDect_withRapidocr): Text detection object.

    Returns:
        List[List[List[float]]]: List of adjusted bounding box coordinates.
    """
    det_results = rapidocr_detector.detect(images)
    
    adjusted_bboxes = []
    for bbox_coords in det_results:
        try:
            adjusted_bbox_coords = adjust_bbox_height(bbox_coords, height_adjustment_factor=0.45)
            adjusted_bboxes.append(adjusted_bbox_coords)
        except ValueError:
            continue  # Skip invalid bbox
    
    return adjusted_bboxes

# def batch_text_detection(images: List[Image.Image], rapidocr_detector: TextDect_withRapidocr) -> List[TextDetectionResult]:
#     # results = []
#     for img in images:
#         # Convert PIL image to numpy array
#         img_array = np.array(img)
#         #  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # Detect text using RapidOCR
#         det_results = rapidocr_detector.detect(img_array)
        
#         bboxes = []
#         for bbox_coords in det_results:
#             try:
#                 adjusted_bbox_coords = adjust_bbox_height(bbox_coords, height_adjustment_factor=0.25)
#                 polygon_box = PolygonBox(polygon=adjusted_bbox_coords)
#                 bboxes.append(polygon_box)
#             except ValueError as e:
#                 continue  # Skip invalid bbox
        
#         # result = TextDetectionResult(
#         #     bboxes=bboxes,
#         #     vertical_lines=[],  # Add logic for detecting vertical lines if needed
#         #     heatmap=None,       # Create a heatmap if needed
#         #     affinity_map=None,  # Create an affinity map if needed
#         #     image_bbox=[0, 0, img.width, img.height]
#         # )
#         # results.append(result)
#     return bboxes