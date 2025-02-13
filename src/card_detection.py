import cv2
from src.utils import detect_objects, apply_nms, get_card_corners

def detect_corners(corner_detector, image, iou_threshold=0.5):
    
    """
    Detect the four corners of the ID card and apply NMS to filter out overlapping bounding boxes.
    Args:
        image: np.array
        iou_threshold (float): NMS threshold.
    Returns:
        tuple: (filtered_bboxes, corner_centers)
    """
    
    print(f"[INFO] Loading image...")
    if image is None:
        print("[ERROR] Can not load image!")
        return None, None

    print("[INFO] Running YOLO model for corner detection...")

    raw_detections = detect_objects(image, corner_detector)

    if not raw_detections:
        print("[ERROR] No corners detected")
        return None, None

    print("[INFO] Raw detected corners:", raw_detections)

    # Apply NMS
    filtered_corners = apply_nms(raw_detections, iou_threshold)

    # Define center of each corner
    corner_centers = get_card_corners(filtered_corners)

    if corner_centers is None:
        return None, None

    return filtered_corners, corner_centers
