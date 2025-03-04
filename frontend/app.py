import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image
import pillow_heif
import tempfile

# Import các module xử lý
from src.config import THRESHOLD
from src.config import CORNER_MODEL_PATH, TEXT_MODEL_PATH, VIETOCR_MODEL_PATH, FACE_DETECTION_MODEL_PATH, FACE_CARD_MODEL_PATH
from src.utils import detect, embed_facenet, compute_similarity
from src.face_embedding import embedding
from src.face_verification import embed_face_cccd, verify_identity
from src.card_detection import detect_corners
from src.transform_card import process_card_transformation
from src.text_detection import detect_text_regions
from src.text_recognition import load_vietocr, extract_text_from_boxes
from src.database import save_embeddings_to_firestore, get_embeddings_from_firestore
st.title("🆔 Face Verification & CCCD Extraction")

# ✅ Cache models để tối ưu
@st.cache_resource
def get_corner_model():
    from ultralytics import YOLO
    return YOLO(CORNER_MODEL_PATH)

@st.cache_resource
def get_text_model():
    from ultralytics import YOLO
    return YOLO(TEXT_MODEL_PATH)

@st.cache_resource
def get_face_model():
    from ultralytics import YOLO
    return YOLO(FACE_DETECTION_MODEL_PATH)

@st.cache_resource
def get_face_card_model():
    from ultralytics import YOLO
    return YOLO(FACE_CARD_MODEL_PATH)

@st.cache_resource
def get_facenet_model():
    from facenet_pytorch import InceptionResnetV1
    return InceptionResnetV1(pretrained="vggface2").eval()

@st.cache_resource
def get_ocr_model():
    return load_vietocr()

# 🏠 Load các model chỉ một lần
corner_model = get_corner_model()
text_model = get_text_model()
face_model = get_face_model()
face_card_model = get_face_card_model()
facenet_model = get_facenet_model()
ocr_model = get_ocr_model()

# 🔄 Session state để kiểm soát hiển thị phần CCCD
if "embeddings_done" not in st.session_state:
    st.session_state.embeddings_done = False
    
# 📌 User nhập ID của họ
user_id = st.text_input("🔑 Nhập User ID:", placeholder="Nhập mã định danh của bạn")
st.write(f"Welcome {user_id} to my web")

# ✅ Upload video để tạo embeddings
st.subheader("🎥 Upload Video để tạo Face Embeddings")
video_file = st.file_uploader("📂 **Chọn video**", type=["mp4", "avi", "mov"])

if video_file is not None and not st.session_state.get("embeddings_done", False):
    st.write("📌 **Đang xử lý video...**")

    # ✅ Lưu file video vào một tệp tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name  # Lấy đường dẫn tệp tạm

    # ✅ Mở video bằng OpenCV
    video_cap = cv2.VideoCapture(temp_video_path)

    if not video_cap.isOpened():
        st.error("❌ Không thể mở video! Hãy thử upload lại.")
        st.stop()
    else:
        st.success("✅ Video đã được mở thành công!")

    # 🟢 Chạy pipeline embedding
    embeddings = embedding(face_model, facenet_model, video_cap)

    if not embeddings:
        st.error("❌ Không thể trích xuất embeddings từ video! Hãy chọn video khác.")
        st.stop()

    save_embeddings_to_firestore(user_id, embeddings)
    
    st.success(f"✅ Đã lưu embeddings thành công cho user `{user_id}`!")
    st.session_state.embeddings_done = True  # ✅ Đánh dấu đã tạo embeddings

    # Xóa message sau khi chạy xong
    st.rerun()

# ✅ Chỉ hiển thị phần CCCD nếu embeddings đã tạo xong
if st.session_state.embeddings_done:
    st.subheader("📸 Upload ảnh CCCD để xác thực")
    uploaded_file = st.file_uploader("📂 **Chọn ảnh CCCD**", type=["jpg", "jpeg", "png", "HEIC"])

    if uploaded_file is not None:
        st.write("📌 **Đang xử lý ảnh CCCD...**")

        # 🟢 Đọc ảnh (hỗ trợ HEIC)
        try:
            if uploaded_file.name.lower().endswith(".heic"):
                heif_image = pillow_heif.open_heif(uploaded_file)
                image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
            else:
                image = Image.open(uploaded_file)
            image = image.convert("RGB")
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc ảnh: {e}")
            st.stop()

        # Chuyển ảnh sang numpy array
        image_np = np.array(image)

        # 🔍 Embed khuôn mặt từ ảnh CCCD
        status = st.empty()  # 👈 Tạo placeholder để xóa trạng thái sau khi chạy xong
        face_embedding_cccd, error = embed_face_cccd(face_card_model, facenet_model, image_np)
        # status.empty()  # ❌ Xóa dòng trạng thái
        if error:
            st.error(error)  # Hiển thị lỗi trên UI
            st.stop()
        
        user_embeddings = get_embeddings_from_firestore(user_id)
        if user_embeddings is None:
            st.error(f"❌ Không tìm thấy embeddings của user `{user_id}`!")
            st.stop()

        # 🔍 So sánh với embeddings từ video
        status = st.empty()
        status.write("🔍 **Đang xác thực danh tính...**")
        verified, max_similarity = verify_identity(face_embedding_cccd, user_embeddings, threshold = THRESHOLD)
        # status.empty()

        st.write(f"📊 **Cosine Similarity:** {max_similarity:.4f}")

        if not verified:
            st.error("❌ **Xác thực thất bại!** Khuôn mặt không khớp.")
            st.stop()

        st.success("✅ **Xác thực thành công!** Tiếp tục trích xuất thông tin CCCD.")

        # 🔍 Nhận diện góc CCCD
        status = st.empty()
        status.write("🔍 **Đang nhận diện 4 góc CCCD...**")
        filtered_corners, corner_centers = detect_corners(corner_model, image_np)
        status.empty()

        if not filtered_corners:
            st.error("🚨 Không nhận diện được đủ 4 góc thẻ CCCD!")
            st.stop()

        # 📐 Transform perspective
        status = st.empty()
        status.write("📐 **Đang xử lý perspective transformation...**")
        transformed_image = process_card_transformation(image_np, corner_centers)
        status.empty()

        if transformed_image is None:
            st.error("🚨 Không thể chuyển đổi góc ảnh!")
            st.stop()

        # 📝 Phát hiện vùng chứa text
        status = st.empty()
        status.write("📝 **Đang phát hiện vùng chứa văn bản...**")
        text_boxes = detect_text_regions(text_model, transformed_image)
        status.empty()

        if not text_boxes:
            st.error("🚨 Không tìm thấy vùng văn bản nào!")
            st.stop()

        # 🔠 Trích xuất văn bản
        status = st.empty()
        status.write("🔠 **Đang nhận diện văn bản...**")
        extracted_texts = extract_text_from_boxes(transformed_image, text_boxes, ocr_model)
        status.empty()

        if not extracted_texts or all(v == "N/A" for v in extracted_texts.values()):
            st.error("🚨 Không nhận diện được văn bản!")
            st.stop()

        # ✨ Hiển thị kết quả
        st.subheader("📋 Thông tin trích xuất:")
        st.write(f"**🆔 Số CCCD:** {extracted_texts.get('id', 'N/A')}")
        st.write(f"**👤 Họ và tên:** {extracted_texts.get('name', 'N/A')}")
        st.write(f"**📅 Ngày sinh:** {extracted_texts.get('birth', 'N/A')}")

        st.success("✅ **Hoàn tất!**")
