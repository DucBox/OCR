import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import matplotlib.pyplot as plt
from src.config import CORNER_MODEL_PATH, TEXT_MODEL_PATH, VIETOCR_MODEL_PATH
from src.card_detection import load_yolo_model, detect_corners
from src.transform_card import perspective_transform
from src.text_detection import detect_text_regions
from src.text_recognition import load_vietocr, extract_text_from_boxes

# Đường dẫn ảnh test (Cập nhật ảnh cụ thể)
IMAGE_PATH = "/Users/ngoquangduc/Desktop/AI_Project/Card_ID/data/raw/IMG_2215.JPG"

def main():
    print("\n🚀 [INFO] Starting Inference...\n")

    # 🔹 Load YOLO model
    print(f"📂 [INFO] Loading corner detection model from: {CORNER_MODEL_PATH}")
    corner_model = load_yolo_model(CORNER_MODEL_PATH)

    print(f"📂 [INFO] Loading text detection model from: {TEXT_MODEL_PATH}")
    text_model = load_yolo_model(TEXT_MODEL_PATH)

    # 🔹 Load image
    print(f"🖼 [INFO] Loading image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ [ERROR] Failed to load image. Check the file path.")
        return

    # 🔹 Detect corners
    print("🔍 [INFO] Running corner detection...")
    filtered_corners, corner_centers = detect_corners(corner_model, image)
    print(f"✅ [INFO] Detected corners: {filtered_corners}")

    if not filtered_corners:
        print("❌ [ERROR] No corners detected. Exiting.")
        return

    # 🔹 Perspective Transform
    print("📐 [INFO] Performing perspective transform...")
    transformed_image = perspective_transform(image, corner_centers)
    if transformed_image is None:
        print("❌ [ERROR] Perspective transformation failed. Exiting.")
        return

    # 🔹 Detect text regions
    print("📝 [INFO] Running text region detection...")
    text_boxes = detect_text_regions(text_model, transformed_image)
    if not text_boxes:
        print("⚠️ [WARNING] No text regions detected.")

    # 🔹 Load VietOCR model
    print(f"📂 [INFO] Loading OCR model from: {VIETOCR_MODEL_PATH}")
    vietocr_detector = load_vietocr()

    # 🔹 Extract text
    if text_boxes:
        print("🔠 [INFO] Running OCR to extract text...")
        extracted_texts = extract_text_from_boxes(transformed_image, text_boxes, vietocr_detector)
    else:
        extracted_texts = {}

    # 🔹 Print extracted information
    print("\n🎯 [RESULT] Extracted Information:")
    print(f"🆔 ID: {extracted_texts.get('id', 'N/A')}")
    print(f"👤 Name: {extracted_texts.get('name', 'N/A')}")
    print(f"📅 Birth: {extracted_texts.get('birth', 'N/A')}")

if __name__ == "__main__":
    main()
