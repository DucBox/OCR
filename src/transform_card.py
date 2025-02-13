import cv2
import numpy as np
from src.utils import transform_perspective

def validate_corners(corners):
    """
    Check there are 4 corners or not

    Args:
        corners (dict): Dict that stores coordinates of corners

    Returns:
        bool: True if there are 4 corners, False in contrast
    """
    required_corners = {"top_left", "top_right", "bottom_left", "bottom_right"}
    return required_corners.issubset(corners.keys())

def process_card_transformation(image, corners):
    """
    Transform card (Perspective transformation)

    Args:
        image (np.array): image
        corners (dict): Dict that stores coordinates of corners

    Returns:
        numpy.ndarray | None: Images after transforming or None
    """
    print(f"[INFO] Loading image...")
    if image is None:
        print("[ERROR] Can not load image")
        return None

    if not validate_corners(corners):
        print("[ERROR] Not enough 4 corners to transform")
        return None

    print("[INFO] Transforming perspective...")
    transformed_image = transform_perspective(image, corners)

    if transformed_image is not None:
        print("[SUCCESS] Perspective transform completed.")
    else:
        print("[ERROR] Transform failed!")

    return transformed_image
