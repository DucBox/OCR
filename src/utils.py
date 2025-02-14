import os
import cv2
import torch
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from src.config import (
    FACE_DETECTION_MODEL_PATH, FACENET_MODEL_PATH, FACE_EMBEDDINGS_PATH, CORNER_MODEL_PATH, TEXT_MODEL_PATH, VIETOCR_MODEL_PATH
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_frames(video):
    """
    Extract first - middle - last frame every second from video.

    Args:
    video_path (str): Video path.

    Returns:
    list: List of frames [(sec, frame_pos, frame)].
    """
    cap = video
    if not cap.isOpened():
        print("[ERROR] Can not open video")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  

    print(f"[INFO] FPS: {fps}, Tổng số frames: {frame_count}, Thời lượng: {duration:.2f} giây")

    # Chọn frame đầu - giữa - cuối mỗi giây
    frame_selection = [0, fps // 2, fps - 1]
    
    frames = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        sec = frame_idx // fps
        frame_pos = frame_idx % fps

        if frame_pos in frame_selection:
            frames.append((sec, frame_pos, frame))

        frame_idx += 1

    cap.release()
    print(f"[INFO] Đã trích xuất {len(frames)} frames.")
    return frames

def detect(image, model):
    """
    Detect faces in an image.

    Args:
    image (np.ndarray): Input image.

    Returns:
    list: List of bboxes (x1, y1, x2, y2).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    bboxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bboxes.append((x1, y1, x2, y2))

    return bboxes

def crop(image, bbox):
    """
    Crop face from image by bbox.

    Args:
    image (np.ndarray): Input image.
    bbox (tuple): (x1, y1, x2, y2) bbox coordinates.

    Returns:
    PIL.Image: Crop face image.
    """
    x1, y1, x2, y2 = bbox
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

def preprocess_img(face_image):
    """
    Preprocess the face image before feeding it into FaceNet.

    Args:
    face_image (PIL.Image): Face image.

    Returns:
    torch.Tensor: Normalized image tensor.

    """
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face_image).unsqueeze(0)

def embed_facenet(face_image, facenet_model):
    """
    Face detection and feature vector generation using FaceNet.

    Args:
    face_image (PIL.Image): Face image.

    Returns:
    numpy.ndarray: Feature vector (512,)
    """
    face_tensor = preprocess_img(face_image).to(device)
    
    with torch.no_grad():
        embedding = facenet_model(face_tensor).cpu().numpy()

    return embedding.squeeze()

def compute_similarity(vec1, vec2):
    """
    Calculates the Cosine similarity between two vectors.

    Args:
    vec1 (numpy.ndarray): First feature vector.
    vec2 (numpy.ndarray): Second feature vector.

    Returns:
    float: Cosine similarity value.
    """
    vec1 = np.array(vec1).squeeze()
    vec2 = np.array(vec2).squeeze()
    
    return cosine_similarity([vec1], [vec2])[0][0]

def save_embeddings(embeddings):
    with open(FACE_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(f"[💾 SAVED] Đã lưu {len(embeddings)} embeddings vào {FACE_EMBEDDINGS_PATH}")

def load_embeddings():
    if not os.path.exists(FACE_EMBEDDINGS_PATH):
        print(f"[ERROR] Không tìm thấy database embeddings: {FACE_EMBEDDINGS_PATH}")
        return None

    with open(FACE_EMBEDDINGS_PATH, "rb") as f:
        embeddings_data = pickle.load(f)

    return embeddings_data

def get_card_corners(corner_bboxes):
    """
    Determine the coordinates of the 4 corners of the card based on the bbox of the corners.

    Args:
    corner_bboxes (dict): Dictionary containing the bbox of the corners.

    Returns:
    dict: Dictionary containing the center coordinates of the 4 corners.
    """
    required_corners = {"top_left", "top_right", "bottom_left", "bottom_right"}
    
    if not required_corners.issubset(corner_bboxes.keys()):
        print("❌ [ERROR] Không đủ 4 góc, yêu cầu user upload ảnh khác.")
        return None

    corner_centers = {
        label: ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2)
        for label, coords in corner_bboxes.items()
    }

    return corner_centers

def transform_perspective(image, corners, output_size=(800, 500)):
    """
    Transform the perspective of the CCCD image to the standard form.

    Args:
    image (numpy.ndarray): Original image.
    corners (dict): The 4 corner coordinates of the CCCD.
    output_size (tuple): The size of the output image.

    Returns:
    numpy.ndarray: Image after transformation, or None if error.

    """
    try:
        if len(corners) != 4:
            raise ValueError("⚠️ Không đủ 4 góc để transform.")

        src_points = np.array([
            corners["top_left"],
            corners["top_right"],
            corners["bottom_left"],
            corners["bottom_right"]
        ], dtype=np.float32)

        dst_points = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [0, output_size[1] - 1],
            [output_size[0] - 1, output_size[1] - 1]
        ], dtype=np.float32)

        # Tính ma trận biến đổi
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(image, M, output_size)

        return transformed_image
    except Exception as e:
        print(f"❌ [ERROR] {e}")
        return None

def extract_text(image, model):
    """
    Recognize text from cropped image.

    Args:
    image (numpy.ndarray): Image containing text.
    model (Predictor): OCR model.

    Returns:
    str: Recognized character string.
    """
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        text = model.predict(image_pil).strip()
        return text
    except Exception as e:
        print(f"❌ [ERROR] Lỗi OCR: {e}")
        return ""

def compute_iou(box1, box2):
    """
    Calculates the IoU (Intersection over Union) between two bounding boxes.

    Args:
    box1 (tuple): (x1, y1, x2, y2) coordinates of the first bbox.
    box2 (tuple): (x1, y1, x2, y2) coordinates of the second bbox.

    Returns:
    float: IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter out duplicate bboxes.

    Args:
    detections (list): List of bboxes [(bbox, confidence, label)].
    iou_threshold (float): IoU threshold to remove bboxes.

    Returns:
    dict: Dictionary {label: bbox} containing the best bboxes for each corner.

    """
    print("[INFO] Applying NMS...")
    
    filtered_detections = {}
    unique_labels = set(label for _, _, label in detections)
    
    for label in unique_labels:
        label_detections = [det for det in detections if det[2] == label]
        label_detections.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo confidence

        selected_boxes = []
        while label_detections:
            best_box = label_detections.pop(0)
            selected_boxes.append(best_box)
            
            label_detections = [
                box for box in label_detections
                if compute_iou(best_box[0], box[0]) < iou_threshold
            ]

        if selected_boxes:
            filtered_detections[label] = selected_boxes[0][0]
    
    print("[INFO] Filtered detections:", filtered_detections)
    return filtered_detections

def detect_objects(image, model):
    """
    Detect objects (CCCD corners, text areas) in images using YOLO.

    Args:
    image (numpy.ndarray): Input image.
    model (YOLO): Loaded YOLO model.

    Returns:
    list: [(bbox, confidence, label), ...]
    """
    results = model(image)
    
    detections = [
        ((int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])),
         float(box.conf[0]),  # Confidence score
         model.names[int(box.cls[0])])  # Label
        for box in results[0].boxes
    ]
    
    return detections