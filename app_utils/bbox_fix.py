
def merge_overlapping_bboxes(bboxes):
    """
    Merges overlapping bounding boxes.

    Args:
        bboxes (List[List[int]]): List of bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
        List[List[int]]: List of merged bounding boxes.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    # Convert bounding boxes to Shapely boxes
    shapely_boxes = [box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bboxes]

    # Merge overlapping boxes using unary_union
    merged = unary_union(shapely_boxes)

    # Ensure merged is a list of polygons
    if merged.geom_type == 'Polygon':
        merged_polygons = [merged]
    elif merged.geom_type == 'MultiPolygon':
        merged_polygons = list(merged.geoms)  # Use .geoms attribute here
    else:
        merged_polygons = []

    # Convert merged polygons back to bounding boxes
    merged_bboxes = []
    for poly in merged_polygons:
        x_min, y_min, x_max, y_max = map(int, poly.bounds)
        merged_bboxes.append([x_min, y_min, x_max, y_max])

    # Sort merged bounding boxes by their y_min to maintain top-down order
    merged_bboxes.sort(key=lambda bbox: bbox[1])

    return merged_bboxes





def is_mrz(text):
    if not text:
        return False
    # Check if a significant portion of the text consists of '<' or '>'
    count_special_chars = text.count('<') + text.count('>')
    return (count_special_chars / len(text)) > 0.2
