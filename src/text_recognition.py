import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
from src.config import VIETOCR_MODEL_PATH

def load_vietocr():
    """
    
    Load mô hình VietOCR để nhận diện văn bản.

    Returns:
        Predictor: Mô hình VietOCR đã load.
    """
    print(f"[INFO] Loading VietOCR model from {VIETOCR_MODEL_PATH}...")

    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = VIETOCR_MODEL_PATH  # Load model từ file
    config['device'] = 'cpu'  # Nếu có GPU, đổi thành 'cuda'

    return Predictor(config)

def extract_text_from_boxes(image, boxes, detector):
    """
    Nhận diện văn bản từ các vùng text crop được.

    Args:
        image (numpy.ndarray): Ảnh CCCD đã transform.
        boxes (dict): Dictionary chứa tọa độ bbox của các vùng text.
        detector (Predictor): Mô hình VietOCR.

    Returns:
        dict: Kết quả OCR {label: text}.
    """
    texts = {}
    for label, (x1, y1, x2, y2) in boxes.items():
        cropped_image = image[y1:y2, x1:x2]  # Crop vùng chứa text
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_pil = Image.fromarray(cropped_image_rgb)

        text = detector.predict(cropped_image_pil).strip()  # Nhận diện text
        texts[label] = text

        print(f"[INFO] Extracted {label}: {text}")

    return texts
