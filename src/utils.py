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
    Trích xuất frame đầu - giữa - cuối mỗi giây từ video.
    
    Args:
        video_path (str): Đường dẫn video.
    
    Returns:
        list: Danh sách frames [(sec, frame_pos, frame)].
    """
    cap = video
    if not cap.isOpened():
        print("[ERROR] Không thể mở video!")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  # Thời lượng video

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
    Phát hiện khuôn mặt trên ảnh.

    Args:
        image (numpy.ndarray): Ảnh đầu vào.

    Returns:
        list: Danh sách bbox (x1, y1, x2, y2).
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
    Crop khuôn mặt từ ảnh theo bbox.

    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        bbox (tuple): (x1, y1, x2, y2) tọa độ bbox.

    Returns:
        PIL.Image: Ảnh khuôn mặt đã crop.
    """
    x1, y1, x2, y2 = bbox
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

def preprocess_img(face_image):
    """
    Tiền xử lý ảnh khuôn mặt trước khi đưa vào FaceNet.

    Args:
        face_image (PIL.Image): Ảnh khuôn mặt.

    Returns:
        torch.Tensor: Tensor ảnh đã chuẩn hóa.
    """
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face_image).unsqueeze(0)

def embed_facenet(face_image, facenet_model):
    """
    Nhận diện khuôn mặt và tạo vector đặc trưng bằng FaceNet.

    Args:
        face_image (PIL.Image): Ảnh khuôn mặt.

    Returns:
        numpy.ndarray: Vector đặc trưng (512,)
    """
    face_tensor = preprocess_img(face_image).to(device)
    
    with torch.no_grad():
        embedding = facenet_model(face_tensor).cpu().numpy()

    return embedding.squeeze()

def compute_similarity(vec1, vec2):
    """
    Tính toán độ tương đồng Cosine giữa hai vector.

    Args:
        vec1 (numpy.ndarray): Vector đặc trưng đầu tiên.
        vec2 (numpy.ndarray): Vector đặc trưng thứ hai.

    Returns:
        float: Giá trị cosine similarity.
    """
    vec1 = np.array(vec1).squeeze()
    vec2 = np.array(vec2).squeeze()
    
    return cosine_similarity([vec1], [vec2])[0][0]

def save_embeddings(embeddings):
    """
    Lưu embeddings vào file.

    Args:
        embeddings (dict): Dictionary chứa các embeddings.
    """
    with open(FACE_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(f"[💾 SAVED] Đã lưu {len(embeddings)} embeddings vào {FACE_EMBEDDINGS_PATH}")

def load_embeddings():
    """
    Tải embeddings đã lưu.

    Returns:
        dict: Dictionary chứa các embeddings.
    """
    if not os.path.exists(FACE_EMBEDDINGS_PATH):
        print(f"[ERROR] Không tìm thấy database embeddings: {FACE_EMBEDDINGS_PATH}")
        return None

    with open(FACE_EMBEDDINGS_PATH, "rb") as f:
        embeddings_data = pickle.load(f)

    return embeddings_data

def get_card_corners(corner_bboxes):
    """
    Xác định tọa độ 4 góc của thẻ dựa vào bbox của các corners.

    Args:
        corner_bboxes (dict): Dictionary chứa bbox của các góc.

    Returns:
        dict: Dictionary chứa tọa độ trung tâm của 4 góc.
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
    Biến đổi phối cảnh ảnh CCCD về dạng chuẩn.

    Args:
        image (numpy.ndarray): Ảnh gốc.
        corners (dict): Tọa độ 4 góc của CCCD.
        output_size (tuple): Kích thước ảnh đầu ra.

    Returns:
        numpy.ndarray: Ảnh sau khi transform, hoặc None nếu lỗi.
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
    Nhận diện văn bản từ ảnh đã crop.

    Args:
        image (numpy.ndarray): Ảnh vùng chứa text.
        model (Predictor): Mô hình OCR.

    Returns:
        str: Chuỗi ký tự nhận diện được.
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
    Tính toán IoU (Intersection over Union) giữa hai bounding box.

    Args:
        box1 (tuple): (x1, y1, x2, y2) tọa độ bbox đầu tiên.
        box2 (tuple): (x1, y1, x2, y2) tọa độ bbox thứ hai.

    Returns:
        float: Giá trị IoU.
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
    Áp dụng Non-Maximum Suppression (NMS) để lọc các bbox bị trùng.

    Args:
        detections (list): Danh sách bbox [(bbox, confidence, label)].
        iou_threshold (float): Ngưỡng IoU để loại bỏ bbox.

    Returns:
        dict: Dictionary {label: bbox} chứa bbox tốt nhất cho mỗi góc.
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
    Phát hiện đối tượng (góc CCCD, vùng văn bản) trên ảnh bằng YOLO.

    Args:
        image (numpy.ndarray): Ảnh đầu vào.
        model (YOLO): Model YOLO đã load.

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