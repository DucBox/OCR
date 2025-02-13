import cv2
import numpy as np
from src.utils import transform_perspective

def validate_corners(corners):
    """
    Kiểm tra xem có đủ 4 góc của CCCD hay không.

    Args:
        corners (dict): Dictionary chứa tọa độ các góc.

    Returns:
        bool: True nếu có đủ 4 góc, False nếu thiếu.
    """
    required_corners = {"top_left", "top_right", "bottom_left", "bottom_right"}
    return required_corners.issubset(corners.keys())

def process_card_transformation(image, corners):
    """
    Biến đổi phối cảnh ảnh CCCD về dạng chuẩn.

    Args:
        image_path (str): Đường dẫn ảnh CCCD.
        corners (dict): Tọa độ 4 góc của CCCD.

    Returns:
        numpy.ndarray | None: Ảnh sau transform hoặc None nếu thất bại.
    """
    print(f"[INFO] Loading image...")
    if image is None:
        print("❌ [ERROR] Không thể load ảnh!")
        return None

    # 🛑 Kiểm tra xem có đủ 4 góc không trước khi transform
    if not validate_corners(corners):
        print("❌ [ERROR] Không đủ 4 góc để thực hiện transform!")
        return None

    print("[INFO] Transforming perspective...")
    transformed_image = transform_perspective(image, corners)

    if transformed_image is not None:
        print("[✅ SUCCESS] Perspective transform completed.")
    else:
        print("❌ [ERROR] Transform failed!")

    return transformed_image
