# 🚀 Face Verification & ID Extraction

## 📌 Introduction
This project is a face verification and ID card information extraction system. It utilizes **Streamlit** for the user interface, **Firestore (Firebase)** for storing embeddings, and **deep learning models** for image processing.

---

## 🛠 Installation

### 🔹 1. Clone the Project
```bash
git clone https://github.com/DucBox/OCR.git
cd OCR
```

### 🔹 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔥 Configuration

### 🔹 1. Database Configuration
All configurations are stored in `config.py`. Update the **database path** before running the application.

- Open `config.py` and set `DATABASE_CONFIG_PATH`.
- **How to get Firestore credentials from Firebase**:
  1. Go to [Firebase Console](https://console.firebase.google.com/).
  2. Navigate to **Project Settings** > **Service Accounts**.
  3. Click **Generate new private key**, and download the JSON file.
  4. Place the JSON file in the `src/` directory and update `DATABASE_CONFIG_PATH`.

```python
DATABASE_CONFIG_PATH = "src/firebase_config.json"
```

### 🔹 2. Model Paths Configuration
Set the model paths in `config.py`:
```python
MODEL_FACE_EMBEDDING = "models/facenet.pth"
MODEL_TEXT_RECOGNITION = "models/vietocr.pth"
...
```

---

## 🚀 Running the Application

### 🔹 1. Run Locally
```bash
streamlit run frontend/app.py
```
Access the application at `http://localhost:8501`

### 🔹 2. Deploy on Streamlit Cloud
1. **Push the code to GitHub**:
   ```bash
   git push origin main
   ```
2. **Go to [Streamlit Cloud](https://share.streamlit.io/) and connect your GitHub repository.**
3. **Add Firestore Credentials to `Secrets` on Streamlit Cloud**:
   - Open **App settings** → **Secrets**
   - Add the following variables:
     ```ini
     [firebase]
     type = "service_account"
     project_id = "your-project-id"
     private_key_id = "your-private-key-id"
     private_key = "-----BEGIN PRIVATE KEY-----\nMIIEv..."
     ```
4. **Deploy and run the application!** 🚀

---

## 🖥️ Web UI
Below is a preview of the application UI:

![Web UI](WebUI.png)

---

## 🛠 Debugging & Troubleshooting

### 🔹 1. Check Environment Variables Locally
```python
import os
print(os.getenv("DATABASE_CONFIG_PATH"))
```

### 🔹 2. Verify Firestore Connection
In Python shell:
```python
from src.database import get_embeddings_from_firestore
print(get_embeddings_from_firestore("test_user"))
```

---

## 📜 License & Author
- 📌 **Author:** Ngo Quang Duc
- 📌 **Contact:** quangducngo0811@gmail.com

