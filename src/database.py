import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import json
import streamlit as st


# 🔥 Kết nối Firestore
# cred = credentials.Certificate("src/face-embeddings-firebase-adminsdk-fbsvc-3ab14b0c36.json") 
# 🟢 Lấy secrets từ Streamlit Cloud
# firebase_secrets = json.loads(st.secrets["firebase"])
st.write(st.secrets["firebase"])
firebase_secrets = st.secrets["firebase"]
# 🔥 Khởi tạo Firebase chỉ khi chưa được init
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_secrets)  # ✅ Truyền dict trực tiếp
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ✅ Hàm lưu embeddings vào Firestore theo user_id
def save_embeddings_to_firestore(user_id, embeddings):
    """
    Lưu embeddings vào Firestore theo user_id.
    
    Args:
        user_id (str): ID của user.
        embeddings (dict): Dictionary chứa embeddings của user (key: frame_id, value: numpy array).
    """
    doc_ref = db.collection("face_embeddings").document(user_id)

    # 🔹 Chuyển numpy array thành list để lưu JSON hợp lệ
    embeddings_serializable = {}

    for k, v in embeddings.items():
        if isinstance(v, np.ndarray):
            embeddings_serializable[k] = v.tolist()  
        else:
            embeddings_serializable[k] = v

    doc_ref.set({"embeddings": embeddings_serializable})  # ✅ Lưu đúng format JSON
    print(f"✅ Đã lưu embeddings cho user `{user_id}` vào Firestore")

# ✅ Hàm lấy embeddings từ Firestore theo user_id
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
        return {k: np.array(v) for k, v in data.items()}  # ✅ Chuyển list về numpy array
    else:
        print(f"❌ Không tìm thấy embeddings cho user `{user_id}`")
        return None
