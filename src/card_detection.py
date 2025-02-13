import cv2
from src.utils import detect_objects, apply_nms, get_card_corners

def detect_corners(corner_detector, image, iou_threshold=0.5):
    """
    Detect 4 góc của CCCD, áp dụng NMS để lọc bbox bị trùng.

    Args:
        image_path (str): Đường dẫn ảnh CCCD.
        iou_threshold (float): Ngưỡng NMS.

    Returns:
        tuple: (filtered_bboxes, corner_centers)
    """
    print(f"[INFO] Loading image...")
    if image is None:
        print("❌ [ERROR] Không thể load ảnh!")
        return None, None

    print("[INFO] Running YOLO model for corner detection...")

    # 🟢 Gọi `detect()` từ utils.py thay vì chạy lại YOLO trực tiếp
    raw_detections = detect_objects(image, corner_detector)

    if not raw_detections:
        print("❌ [ERROR] Không tìm thấy góc nào!")
        return None, None

    print("[INFO] Raw detected corners:", raw_detections)

    # Áp dụng NMS để lọc bbox bị trùng
    filtered_corners = apply_nms(raw_detections, iou_threshold)

    # Xác định tọa độ trung tâm của từng góc
    corner_centers = get_card_corners(filtered_corners)

    if corner_centers is None:
        return None, None

    return filtered_corners, corner_centers
