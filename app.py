import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# App UI
# =========================
st.title("🧠 Brain Tumor Detection System (Multi-Class)")
st.write("Upload an MRI image to detect tumor type.")

# =========================
# Model Download (only)
# =========================
MODEL_PATH = "model.keras"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "112iwtGioquL_BlN7QQknD4jxQPtvmXAE"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

download_model()

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "jpeg", "png"])

class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # =========================
    # Preprocess Image
    # =========================
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # =========================
    # Prediction (SAFE MODE)
    # =========================
    st.warning("⚠ TensorFlow not supported on this deployment. Prediction disabled.")

    st.info("Image is ready for model input (224x224 normalized).")
