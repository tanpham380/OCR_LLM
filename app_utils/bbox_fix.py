import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import cv2

def polygon_to_bbox(polygon):
    """
    Converts a polygon (list of points) into a bounding box.
    Args:
        polygon (List[List[float]]): List of points in the format [[x1, y1], [x2, y2], ...]
    Returns:
        List[int]: Bounding box in the format [x_min, y_min, x_max, y_max]
    """
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def remove_duplicate_bboxes(bboxes):
    unique_bboxes = []
    seen = set()
    for bbox in bboxes:
        bbox_tuple = tuple(bbox)
        if bbox_tuple not in seen:
            unique_bboxes.append(bbox)
            seen.add(bbox_tuple)
    return unique_bboxes

def merge_overlapping_bboxes(bboxes):
    """
    Merges overlapping bounding boxes.
    Args:
        bboxes (List[List[int]]): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
    Returns:
        List[List[int]]: List of merged bounding boxes.
    """
    # Convert bounding boxes to Shapely boxes
    shapely_boxes = [box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bboxes]

    # Merge overlapping boxes using unary_union
    merged = unary_union(shapely_boxes)

    # Ensure merged is a list of polygons
    merged_polygons = [merged] if merged.geom_type == 'Polygon' else list(merged.geoms)
    
    # Convert merged polygons back to bounding boxes
    merged_bboxes = []
    for poly in merged_polygons:
        x_min, y_min, x_max, y_max = map(int, poly.bounds)
        merged_bboxes.append([x_min, y_min, x_max, y_max])

    # Sort merged bounding boxes by their y_min to maintain top-down order
    merged_bboxes.sort(key=lambda bbox: bbox[1])

    return merged_bboxes
def crop_and_merge_images(img, bboxes):
    """
    Crops and merges images based on bounding boxes.
    
    Args:
        img (np.ndarray): The input image.
        bboxes (List[List[int]]): List of bounding box coordinates.
    
    Returns:
        np.ndarray: Merged image for OCR.
    """
    # Determine if the input image is grayscale or color
    if len(img.shape) == 2:
        # Grayscale image
        channels = 1
    else:
        # Color image
        channels = img.shape[2]

    # Crop each image based on the bounding boxes
    cropped_images = [img[y_min:y_max, x_min:x_max] for x_min, y_min, x_max, y_max in bboxes]

    # Calculate the total width and height for the merged image
    total_width = max([cropped_img.shape[1] for cropped_img in cropped_images])
    total_height = sum([cropped_img.shape[0] for cropped_img in cropped_images])

    # Create the merged image with the appropriate number of channels
    if channels == 1:
        merged_image = np.zeros((total_height, total_width), dtype=img.dtype)
    else:
        merged_image = np.zeros((total_height, total_width, channels), dtype=img.dtype)

    # Paste each cropped image into the merged image
    current_y = 0
    for cropped_img in cropped_images:
        height, width = cropped_img.shape[:2]
        if channels == 1:
            merged_image[current_y:current_y + height, :width] = cropped_img
        else:
            merged_image[current_y:current_y + height, :width, :] = cropped_img
        current_y += height

    return merged_image



def is_mrz(text):
    if not text:
        return False
    # Check if a significant portion of the text consists of '<' or '>'
    count_special_chars = text.count('<') + text.count('>')
    return (count_special_chars / len(text)) > 0.2
