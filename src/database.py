import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import json
import streamlit as st

# 🟢 Lấy secrets từ Streamlit Cloud
firebase_secrets = st.secrets["firebase"]

# 🔥 Convert AttrDict về Dictionary
firebase_secrets_dict = dict(firebase_secrets)

# 🔥 Khởi tạo Firebase chỉ khi chưa được init
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_secrets_dict)  # ✅ Truyền dict đúng kiểu
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 🟢 Kiểm tra kết nối Firestore
try:
    test_doc_ref = db.collection("test").document("streamlit_check")
    test_doc_ref.set({"status": "OK"})
    st.success("✅ Firestore kết nối thành công!")
except Exception as e:
    st.error(f"❌ ERROR: Firestore không hoạt động!\n{e}")
    
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
