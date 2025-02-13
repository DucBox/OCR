import cv2
from src.utils import detect_objects, apply_nms

def filter_text_boxes(text_boxes):
    """
    Lọc bỏ các vùng không cần thiết (các title như 'id_title', 'name_title', 'birth_title').
    Chỉ giữ lại 'id', 'name', 'birth'.

    Args:
        text_boxes (dict): Dictionary chứa bbox và labels.

    Returns:
        dict: Dictionary chứa bbox của các vùng chứa text cần thiết.
    """
    valid_labels = {"id", "name", "birth"}
    filtered_boxes = {label: coords for label, coords in text_boxes.items() if label in valid_labels}

    print("[INFO] Filtered text fields:", filtered_boxes)
    return filtered_boxes

def detect_text_regions(text_detector, image, iou_threshold=0.5):
    """
    Nhận diện các vùng chứa văn bản trên ảnh CCCD bằng YOLO.

    Args:
        image_path (str): Đường dẫn ảnh CCCD.
        iou_threshold (float): Ngưỡng NMS.

    Returns:
        dict: Dictionary chứa tọa độ bbox của các vùng text.
    """
    print(f"[INFO] Loading image...")
    if image is None:
        print("❌ [ERROR] Không thể load ảnh!")
        return None

    print("[INFO] Running YOLO model for text detection...")

    # 🟢 Gọi `detect_objects()` từ `utils.py` để lấy danh sách bbox, confidence, label
    raw_detections = detect_objects(image, text_detector)

    if not raw_detections:
        print("❌ [ERROR] Không tìm thấy vùng chứa văn bản!")
        return None

    print("[INFO] Raw detected text regions:", raw_detections)

    # 🟢 Áp dụng NMS để lọc bbox trùng lặp
    filtered_text_boxes = apply_nms(raw_detections, iou_threshold)

    # 🟢 Lọc bỏ các class title như 'id_title', 'name_title'
    return filter_text_boxes(filtered_text_boxes)
