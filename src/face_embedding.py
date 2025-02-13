import os
from src.utils import extract_frames, detect, crop, embed_facenet, save_embeddings
from src.config import FACE_EMBEDDINGS_PATH

def embedding(face_detector, facenet_model, video):
    """
    Pipeline xử lý video:
    1. Trích xuất frames đầu-giữa-cuối mỗi giây.
    2. Phát hiện khuôn mặt trên mỗi frame.
    3. Crop và embed khuôn mặt bằng FaceNet.
    4. Lưu embeddings vào database.

    Args:
        video_path (str): Đường dẫn video đầu vào.

    Returns:
        dict: Dictionary chứa embeddings của các khuôn mặt.
    """
    print(f"[INFO] Bắt đầu xử lý video")

    # 1️⃣ Trích xuất frames từ video
    frames = extract_frames(video)
    if not frames:
        print("[ERROR] Không có frame nào được trích xuất!")
        return {}

    embeddings = {}

    # 2️⃣ Xử lý từng frame
    for sec, frame_pos, frame in frames:
        # 3️⃣ Phát hiện khuôn mặt
        bboxes = detect(frame, face_detector)
        if not bboxes:
            print(f"[WARNING] Không tìm thấy khuôn mặt tại giây {sec}")
            continue

        for i, bbox in enumerate(bboxes):
            # 4️⃣ Crop khuôn mặt
            face_image = crop(frame, bbox)
            if face_image is None:
                continue

            # 5️⃣ Embed bằng FaceNet
            face_embedding = embed_facenet(face_image, facenet_model)

            # 6️⃣ Lưu vào dictionary
            key = f"frame_{sec}_{frame_pos}_{i}"
            embeddings[key] = face_embedding

    # 7️⃣ Lưu embeddings vào file
    # if embeddings:
    #     save_embeddings(embeddings)
    #     print(f"[INFO] Đã lưu {len(embeddings)} embeddings vào {FACE_EMBEDDINGS_PATH}")
    # else:
    #     print("[WARNING] Không có embeddings nào được lưu!")

    return embeddings
