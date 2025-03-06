import os

# FOLDER PATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings") 

# MODEL
CORNER_MODEL_PATH = os.path.join(MODELS_DIR, "card_detect.pt")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_recog.pt")
VIETOCR_MODEL_PATH = os.path.join(MODELS_DIR, "transformerocr.pth")
FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "head_detect.pt")  
FACE_CARD_MODEL_PATH = os.path.join(MODELS_DIR, "face_card_detect.pt")
FACENET_MODEL_PATH = "vggface2" 
FACE_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")  # Lưu embeddings

FRAME_SELECTION_RATIO = 10

THRESHOLD = 0.7

DATABASE_CONFIG_PATH = '/Users/ngoquangduc/Desktop/AI_Project/Card_ID/src/face-embeddings-firebase-adminsdk-fbsvc-3ab14b0c36.json'

