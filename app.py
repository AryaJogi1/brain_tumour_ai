import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model():
    MODEL_PATH = "model.keras"

    if not os.path.exists(MODEL_PATH):
        file_id = "112iwtGioquL_BlN7QQknD4jxQPtvmXAE"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# UI
# =========================
st.title("🧠 Brain Tumor Detection System (Multi-Class)")
st.write("Upload an MRI image to detect tumor type.")

uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "jpeg", "png"])

# Class names (IMPORTANT)
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # =========================
    # Preprocess
    # =========================
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # =========================
    # Prediction
    # =========================
    prediction = model.predict(img)

    result = np.argmax(prediction)
    confidence = np.max(prediction)

    # =========================
    # Output
    # =========================
    st.write(f"Confidence Score: {confidence:.4f}")
    st.success(f"Prediction: {class_names[result]}")