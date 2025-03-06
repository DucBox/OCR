import firebase_admin
from src.config import DATABASE_CONFIG_PATH
from firebase_admin import credentials, firestore
import numpy as np
import json
import streamlit as st
import os

is_streamlit_cloud = False  

try:
    if hasattr(st, "secrets") and st.secrets:
        print(" Đang chạy trên **Streamlit Cloud**")
        is_streamlit_cloud = "firebase" in st.secrets
except (AttributeError, FileNotFoundError):
    print("💻 Đang chạy trên **Local**")
    is_streamlit_cloud = False

if is_streamlit_cloud:
    print(" Đang chạy trên **Streamlit Cloud** - Dùng `st.secrets`")
    firebase_secrets = dict(st.secrets["firebase"])

    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_secrets)
        firebase_admin.initialize_app(cred)

else:
    print("💻 Đang chạy trên **Local** - Dùng file JSON")

    # 🔥 Đường dẫn file JSON
    firebase_config_path = DATABASE_CONFIG_PATH

    if os.path.exists(firebase_config_path):
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_config_path)
            firebase_admin.initialize_app(cred)
    else:
        print(f" Không tìm thấy Firebase config tại: {firebase_config_path}")
        exit(1)  

db = firestore.client()

try:
    test_doc_ref = db.collection("test").document("streamlit_check")
    test_doc_ref.set({"status": "OK"})
    st.success(" Firestore kết nối thành công!")
except Exception as e:
    st.error(f" ERROR: Firestore không hoạt động!\n{e}")
    
def save_embeddings_to_firestore(user_id, embeddings):
    """
    Lưu embeddings vào Firestore theo user_id.
    
    Args:
        user_id (str): ID của user.
        embeddings (dict): Dictionary chứa embeddings của user (key: frame_id, value: numpy array).
    """
    doc_ref = db.collection("face_embeddings").document(user_id)

    embeddings_serializable = {}

    for k, v in embeddings.items():
        if isinstance(v, np.ndarray):
            embeddings_serializable[k] = v.tolist()  
        else:
            embeddings_serializable[k] = v

    doc_ref.set({"embeddings": embeddings_serializable})  
    print(f" Đã lưu embeddings cho user `{user_id}` vào Firestore")

def get_embeddings_from_firestore(user_id):
    """
    Lấy embeddings từ Firestore theo user_id.
    
    Args:
        user_id (str): ID của user.

    Returns:
        dict: Dictionary chứa embeddings (numpy array).
    """
    doc_ref = db.collection("face_embeddings").document(user_id)
    doc = doc_ref.get()
    
    if doc.exists:
        data = doc.to_dict()["embeddings"]
        return {k: np.array(v) for k, v in data.items()}  
    else:
        print(f" Không tìm thấy embeddings cho user `{user_id}`")
        return None
