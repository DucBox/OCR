import cv2
from ultralytics import YOLO
from src.utils import detect, crop, embed_facenet, load_embeddings, compute_similarity
from src.config import (
    FACE_DETECTION_MODEL_PATH, FACENET_MODEL_PATH
)

def embed_face_cccd(face_detector, facenet_model, image):
    """
    Detect and crop faces from CCCD images.

    Args:
    image(np.array): CCCD image.

    Returns:
    n[].ndarray | None: Feature vector if face found, None if no face.

    """
    # 1️⃣ Read CCCD image
    if image is None:
        print("[ERROR] Can not load image!")
        return None, "❌ Không thể đọc ảnh CCCD! Hãy chụp lại ảnh rõ nét hơn."

    # 2️⃣ Detect face
    bboxes = detect(image, face_detector)
    if not bboxes:
        print("[ERROR] No face detected")
        return None, "❌ Không tìm thấy khuôn mặt trong ảnh CCCD! Hãy chụp lại với góc nhìn rõ hơn."

    # 3️⃣ Crop face
    face_image = crop(image, bboxes[0])
    if face_image is None:
        print("[ERROR] Can not crop face")
        return None, "❌ Không thể crop khuôn mặt từ ảnh CCCD! Hãy chụp lại với ánh sáng tốt hơn."

    # 4️⃣ Embed face
    face_embedding = embed_facenet(face_image, facenet_model)

    if face_embedding is None:
        return None, "❌ Không thể tạo vector embedding từ khuôn mặt!"

    return face_embedding, None  

def verify_identity(face_embedding, embedding_data, threshold=0.7):
    """
    Verify identity by comparing the face on the CCCD with the database embeddings.

    Args:
    threshold (float): Cosine similarity threshold to confirm identity.

    Returns:
    tuple (bool, float): (Authentication successful or not, Highest cosine similarity value)

    """
    print("[INFO] Extract face from CCCD...")
    
    # 1️⃣ Extract vector feature
    if face_embedding is None:
        return False, 0.0

    # 2️⃣ Load embeddings from database extracted from video
    if embedding_data is None:
        print("[ERROR] Database is empty")
        return False, 0.0

    # 3️⃣ Compare
    max_similarity = 0.0
    best_match = None

    for key, saved_vector in embedding_data.items():
        similarity = compute_similarity(face_embedding, saved_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = key

    print(f"[INFO] Highest similarity: {max_similarity:.4f} (Best match: {best_match})")

    return max_similarity >= threshold, max_similarity
