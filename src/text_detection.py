import cv2
from src.utils import detect_objects, apply_nms

def filter_text_boxes(text_boxes):
    """
    Filter out unnecessary fields (titles like 'id_title', 'name_title', 'birth_title')
    Keep only 'id', 'name', 'birth'

    Args:
    text_boxes (dict): Dictionary containing bboxes and labels

    Returns:
    dict: Dictionary containing bboxes of the required text fields
    """
    
    valid_labels = {"id", "name", "birth"}
    filtered_boxes = {label: coords for label, coords in text_boxes.items() if label in valid_labels}

    print("[INFO] Filtered text fields:", filtered_boxes)
    return filtered_boxes

def detect_text_regions(text_detector, image, iou_threshold=0.5):
    """
    Identify text areas on CCCD images using YOLO.

    Args:
    image(np.array): CCCD image 
    iou_threshold (float): NMS threshold

    Returns:
    dict: Dictionary containing bbox coordinates of text areas.
    """
    
    print(f"[INFO] Loading image...")
    if image is None:
        print("[ERROR] Can not load image")
        return None

    print("[INFO] Running YOLO model for text detection...")

    raw_detections = detect_objects(image, text_detector)

    if not raw_detections:
        print("[ERROR] No text regions detected")
        return None

    print("[INFO] Raw detected text regions:", raw_detections)

    # Apply NMS
    filtered_text_boxes = apply_nms(raw_detections, iou_threshold)

    # Filter out class 'id_title', 'name_title'
    return filter_text_boxes(filtered_text_boxes)
