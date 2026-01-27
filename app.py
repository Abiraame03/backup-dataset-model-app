import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import os
import time
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# --- I. Configuration ---
st.set_page_config(page_title="Dyslexia Severity Analyzer", layout="wide")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"
GLOBAL_THRESHOLD = 0.51  # Your requested threshold
IMG_SIZE_DL = (160, 160)

PUZZLES = {
    "Beginner (5-7)": {
        1: "Draw the letters b and d slowly.",
        2: "Write the word CAT in large letters.",
        3: "Write the sentence: The sun is hot."
    },
    "Advanced (8-12)": {
        1: "Draw the letters p, q, b, and d.",
        2: "Write the word MOUNTAIN clearly.",
        3: "Write: The quick brown fox jumps over the lazy dog."
    }
}

# --- II. Model Loading ---
@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except Exception as e:
            st.warning(f"DL Model could not load: {e}")
    return rf, dl

rf_model, dl_model = load_models()

# --- III. Logic ---

def get_severity(prob):
    """Classifies risk based on the 51% threshold."""
    if prob < GLOBAL_THRESHOLD:
        return "Normal", "green", "✅"
    elif GLOBAL_THRESHOLD <= prob < 0.65:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.65 <= prob < 0.85:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def extract_features(img):
    """Extracts HOG features for the RF model."""
    img_res = cv2.resize(img, (64, 64))
    features = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    # Adding geometric placeholders to match your 2011-length feature vector
    placeholders = [np.var(img_res), np.mean(img_res), 0, 0]
    return np.concatenate([features, placeholders]).reshape(1, -1)

def predict_all(gray_img, use_dl=True):
    rf_p, dl_p = 0.0, 0.0
    if rf_model:
        rf_p = rf_model.predict_proba(extract_features(gray_img))[0][1]
    if dl_model and use_dl:
        rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(cv2.resize(rgb, IMG_SIZE_DL)/255.0, axis=0)
        dl_p = float(dl_model.predict(inp, verbose=0)[0][0])
    
    # Combined score (Weighted)
    return (rf_p * 0.4 + dl_p * 0.6) if use_dl else rf_p

# --- IV. UI Tabs ---

if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'results': []})

t1, t2 = st.tabs(["✍️ Assessment Canvas", "📤 Upload Sample"])

with t1:
    if st.session_state.stage <= 3:
        age = st.sidebar.slider("Age", 5, 12, 7)
        task = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"][st.session_state.stage]
        
        st.subheader(f"Task {st.session_state.stage}")
        st.info(f"Instruction: {task}")
        
        canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=700, key=f"v{st.session_state.stage}")
        
        if st.button(f"Submit Task {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                score = predict_all(gray, use_dl=(st.session_state.stage >= 2))
                st.session_state.results.append(score)
                st.session_state.stage += 1
                st.rerun()
    else:
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score)
        st.header(f"{icon} Final Assessment")
        st.markdown(f"### Overall Result: :{color}[{label}]")
        st.write(f"Average Probability: **{avg_score*100:.1f}%**")
        if st.button("Reset"):
            st.session_state.update({'stage': 1, 'results': []})
            st.rerun()

with t2:
    st.header("Upload Handwriting Image")
    up = st.file_uploader("Upload a clear image of a sentence", type=['png', 'jpg', 'jpeg'])
    if up:
        img = Image.open(up).convert('L')
        st.image(img, width=400)
        if st.button("Analyze Upload"):
            score = predict_all(np.array(img), use_dl=True)
            label, color, icon = get_severity(score)
            st.markdown(f"## {icon} Detection: :{color}[{label}]")
            st.progress(score)
            st.write(f"System Certainty: {score*100:.1f}%")
