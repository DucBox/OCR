import os

# 🏠 Định nghĩa thư mục gốc & thư mục models
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")  # 📂 Nơi lưu embeddings

# 📌 Định nghĩa đường dẫn các model
CORNER_MODEL_PATH = os.path.join(MODELS_DIR, "card_detect.pt")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_recog.pt")
VIETOCR_MODEL_PATH = os.path.join(MODELS_DIR, "transformerocr.pth")
FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "head_detect.pt")  # 🔥 Model detect face
FACENET_MODEL_PATH = "vggface2"  # Model pretrain trong `facenet_pytorch`
FACE_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")  # Lưu embeddings

FRAME_SELECTION_RATIO = 10

THRESHOLD = 0.7

DATABASE_CONFIG_PATH = 'path_to_json'

