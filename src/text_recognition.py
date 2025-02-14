import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
from src.config import VIETOCR_MODEL_PATH

def load_vietocr():
    """
    Load VietOCR model to recognize text.

    Returns:
    Predictor: VietOCR model loaded.

    """
    
    print(f"[INFO] Loading VietOCR model from {VIETOCR_MODEL_PATH}...")

    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = VIETOCR_MODEL_PATH 
    config['device'] = 'cpu'  

    return Predictor(config)

def extract_text_from_boxes(image, boxes, detector):
    """
    Recognize text from cropped text regions.

    Args:
    image (np.ndarray): Transformed CCCD image.
    boxes (dict): Dictionary containing bbox coordinates of text regions.
    detector (Predictor): VietOCR model.

    Returns:
    dict: OCR result {label: text}.
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
