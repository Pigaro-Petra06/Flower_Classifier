import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ======================
# CONFIG
# ======================

APP_TITLE = "🌸 Flower Image Classifier"
IMG_SIZE = 128

# Change these paths if your files are stored elsewhere
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "flower_image_model.h5")
LABELS_PATH = os.path.join(BASE_PATH, "class_names.json")

# Optional: use the original dataset folder to recover class names
DATASET_TRAIN_DIR = os.path.join(BASE_PATH, "flowers-dataset", "train")


# ======================
# HELPERS
# ======================

def _load_class_names() -> list[str]:
    """
    Priority:
    1) class_names.json (recommended)
    2) folders inside flowers-dataset/train
    """
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and labels:
            return labels

    if os.path.isdir(DATASET_TRAIN_DIR):
        labels = sorted(
            [d for d in os.listdir(DATASET_TRAIN_DIR)
             if os.path.isdir(os.path.join(DATASET_TRAIN_DIR, d))]
        )
        if labels:
            return labels

    # Fallback only; make sure this matches training order
    return [
        "daisy", "dandelion", "rose", "sunflower", "tulip"
    ]


@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            f"Put 'flower_image_model.h5' in the same folder as this app."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = _load_class_names()
    return model, labels


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert uploaded image to model-ready tensor:
    - resize to 128x128
    - convert to RGB
    - preprocess for ResNet50
    - shape: (1, 128, 128, 3)
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_image(model, labels, image: Image.Image):
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"Class {pred_idx}"
    return pred_label, probs


# ======================
# UI
# ======================

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")
st.title(APP_TITLE)
st.write("Upload a flower image and the app will predict the class.")

try:
    model, labels = load_model_and_labels()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.sidebar.header("Model Info")
st.sidebar.write(f"**Model file:** `{os.path.basename(MODEL_PATH)}`")
st.sidebar.write(f"**Input size:** `{IMG_SIZE} x {IMG_SIZE}`")
st.sidebar.write(f"**Classes:** {len(labels)}")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Image")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        try:
            pred_label, probs = predict_image(model, labels, image)
            pred_conf = float(np.max(probs))

            st.success(f"Predicted label: **{pred_label}**")
            st.metric("Confidence", f"{pred_conf:.2%}")

            st.write("### Probabilities")
            for label, prob in sorted(zip(labels, probs), key=lambda x: x[1], reverse=True):
                st.write(f"- **{label}**: {prob:.4f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

    with st.expander("Image preprocessing preview"):
        preview = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        st.image(preview, caption=f"Resized to {IMG_SIZE}×{IMG_SIZE}", width=220)

else:
    st.info("Upload a flower image to see the prediction.")
