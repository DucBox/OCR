import cv2
from ultralytics import YOLO
from src.utils import detect, crop, embed_facenet, load_embeddings, compute_similarity
from src.config import (
    FACE_DETECTION_MODEL_PATH, FACENET_MODEL_PATH
)

def embed_face_cccd(face_detector, facenet_model, image):
    """
    Nhận diện và crop khuôn mặt từ ảnh CCCD.

    Args:
        cccd_image_path (str): Đường dẫn ảnh CCCD.

    Returns:
        numpy.ndarray | None: Vector đặc trưng nếu tìm thấy khuôn mặt, None nếu không có khuôn mặt.
    """
    # 1️⃣ Đọc ảnh CCCD
    if image is None:
        print("[ERROR] Không thể đọc ảnh CCCD!")
        return None

    # 2️⃣ Phát hiện khuôn mặt
    bboxes = detect(image, face_detector)
    if not bboxes:
        print("[ERROR] Không tìm thấy khuôn mặt trong CCCD!")
        return None

    # 3️⃣ Crop khuôn mặt (do CCCD chỉ có 1 khuôn mặt, lấy bbox đầu tiên)
    face_image = crop(image, bboxes[0])
    if face_image is None:
        print("[ERROR] Không thể crop khuôn mặt từ ảnh CCCD!")
        return None

    # 4️⃣ Embed khuôn mặt
    return embed_facenet(face_image, facenet_model)

def verify_identity(face_embedding, embedding_data, threshold=0.7):
    """
    Xác thực danh tính bằng cách so sánh khuôn mặt trên CCCD với database embeddings.

    Args:
        threshold (float): Ngưỡng cosine similarity để xác nhận danh tính.

    Returns:
        tuple (bool, float): (Xác thực thành công hay không, Giá trị cosine similarity cao nhất)
    """
    print("[INFO] Trích xuất khuôn mặt từ CCCD...")
    
    # 1️⃣ Lấy vector đặc trưng từ CCCD
    if face_embedding is None:
        return False, 0.0

    # 2️⃣ Tải database embeddings từ video
    if embedding_data is None:
        print("[ERROR] Không có embeddings nào để so sánh!")
        return False, 0.0

    # 3️⃣ So sánh với tất cả embeddings
    max_similarity = 0.0
    best_match = None

    for key, saved_vector in embedding_data.items():
        similarity = compute_similarity(face_embedding, saved_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = key

    print(f"[INFO] Highest similarity: {max_similarity:.4f} (Best match: {best_match})")

    return max_similarity >= threshold, max_similarity
