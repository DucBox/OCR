import os
from src.utils import extract_frames, detect, crop, embed_facenet, save_embeddings
from src.config import FACE_EMBEDDINGS_PATH

def embedding(face_detector, facenet_model, video):
    """
    Video processing pipeline:
    1. Extract first-mid-last frames every second.
    2. Detect faces in each frame.
    3. Crop and embed faces using FaceNet.
    4. Save embeddings to database.

    Args:
    video: Input video

    Returns:
    dict: Dictionary containing embeddings of faces
    """
    print(f"[INFO] Start processing Video")

    # 1️⃣ Extract frames from video
    frames = extract_frames(video)
    if not frames:
        print("[ERROR] No frame extracted")
        return {}

    embeddings = {}

    # 2️⃣ Process each frame
    for sec, frame_pos, frame in frames:
        # 3️⃣ Detect Face
        bboxes = detect(frame, face_detector)
        if not bboxes:
            print(f"[WARNING] No face detected at {sec}")
            continue

        for i, bbox in enumerate(bboxes):
            # 4️⃣ Crop Face
            face_image = crop(frame, bbox)
            if face_image is None:
                continue

            # 5️⃣ Embed using FaceNet
            face_embedding = embed_facenet(face_image, facenet_model)

            # 6️⃣ Save into dictionary
            key = f"frame_{sec}_{frame_pos}_{i}"
            embeddings[key] = face_embedding

    return embeddings
