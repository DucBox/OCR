import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import json
import streamlit as st

import firebase_admin
from firebase_admin import credentials, firestore
import json
import streamlit as st

# 🟢 DEBUG: Kiểm tra Streamlit secrets
st.write("🔍 DEBUG: Firebase Secrets Type:", type(st.secrets["firebase"]))

try:
    # 🔥 Chuyển AttrDict về dict
    firebase_secrets = json.loads(json.dumps(st.secrets["firebase"]))
    
    # 🟢 DEBUG: Kiểm tra dữ liệu đã convert
    st.write("✅ DEBUG: Firebase Secrets Converted Type:", type(firebase_secrets))

    # 🔥 Kiểm tra xem có key `private_key` không
    if "private_key" not in firebase_secrets:
        st.error("❌ ERROR: `private_key` không có trong secrets! Kiểm tra lại cấu hình Streamlit.")
        st.stop()

    # 🔥 Kiểm tra format của `private_key`
    if not firebase_secrets["private_key"].startswith("-----BEGIN PRIVATE KEY-----"):
        st.error("❌ ERROR: `private_key` format sai! Xem lại cách nhập vào Streamlit secrets.")
        st.write("🔍 private_key hiện tại:", firebase_secrets["private_key"])  # 🟢 Debug giá trị
        st.stop()

    # 🔥 Khởi tạo Firebase nếu chưa được init
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_secrets)  # ✅ Truyền dict đã sửa lỗi
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    st.write("✅ Firebase kết nối thành công!")  # 🟢 Debug thành công

except Exception as e:
    st.error(f"❌ ERROR: Firebase init thất bại: {e}")
    st.stop()

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
