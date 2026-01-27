import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import json
import time
import os
import streamlit.components.v1 as components
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas

# --- I. Configuration and Model Loading ---
st.set_page_config(page_title="Dual-Model Dyslexia Analyzer", layout="wide")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5" 
IMG_SIZE_DL = (160, 160)

# User-requested threshold
GLOBAL_THRESHOLD = 0.51 

PUZZLES = {
    "Beginner (5-7)": {
        1: "Draw the letters b and d slowly and clearly.",
        2: "Write the word CAT using large capital letters.",
        3: "Write the sentence: The sun is hot."
    },
    "Advanced (8-12)": {
        1: "Draw the letters p, q, b, and d in a single row.",
        2: "Write the word MOUNTAIN in your best handwriting.",
        3: "Write: The quick brown fox jumps over the lazy dog."
    }
}

@st.cache_resource
def load_all_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except: pass
    return rf, dl

rf_model, dl_model = load_all_models()

# --- II. Logic Functions ---

def speak_task(text):
    components.html(f"<script>var msg = new SpeechSynthesisUtterance('{text}'); window.speechSynthesis.speak(msg);</script>", height=0)

def get_severity(prob):
    """Maps the probability to a severity label."""
    if prob < 0.51: return "Normal", "green"
    elif 0.51 <= prob < 0.65: return "Mild Dyslexia", "blue"
    elif 0.65 <= prob < 0.85: return "Moderate Dyslexia", "orange"
    else: return "Severe Dyslexia", "red"

def extract_rf_features(img, img_size=64):
    img_res = cv2.resize(img, (img_size, img_size))
    hog_feat = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    edges = cv2.Canny(img_res, 80, 160)
    edge_density = np.sum(edges) / (np.sum(img_res > 0) + 1)
    return np.concatenate([hog_feat, [edge_density, np.var(img_res), 0, 0]])

def run_prediction(gray_img, run_dl=True):
    rf_prob, dl_prob = 0.0, 0.0
    
    if rf_model:
        feats = extract_rf_features(gray_img).reshape(1, -1)
        rf_prob = rf_model.predict_proba(feats)[0][1]
        
    if dl_model and run_dl:
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        dl_input = np.expand_dims(cv2.resize(rgb_img, IMG_SIZE_DL)/255.0, axis=0)
        dl_prob = float(dl_model.predict(dl_input, verbose=0)[0][0])
    
    # Combined score (weighted average)
    final_score = (rf_prob * 0.4) + (dl_prob * 0.6) if run_dl else rf_prob
    return final_score

# --- III. Interface ---

if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'scores': [], 'spoken': False})

tab1, tab2 = st.tabs(["✍️ Audio Assessment", "📤 File Upload"])

with tab1:
    if st.session_state.stage <= 3:
        u_age = st.sidebar.slider("Age", 5, 12, 7)
        bracket = "Beginner (5-7)" if u_age <= 7 else "Advanced (8-12)"
        task_text = PUZZLES[bracket][st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_task(task_text)
            st.session_state.spoken = True

        st.subheader(f"Task {st.session_state.stage}: {task_text}")
        canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=800, key=f"c{st.session_state.stage}")

        if st.button(f"Submit Task {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                score = run_prediction(gray, run_dl=(st.session_state.stage >= 2))
                st.session_state.scores.append(score)
                st.session_state.stage += 1
                st.session_state.spoken = False
                st.rerun()

    elif st.session_state.stage > 3:
        avg_score = np.mean(st.session_state.scores)
        label, color = get_severity(avg_score)
        
        st.header("🏁 Assessment Result")
        if label == "Normal":
            st.balloons()
            st.success(f"### Result: {label}")
        else:
            st.markdown(f"### Result: :{color}[{label}]")
        
        st.metric("Total Risk Probability", f"{avg_score*100:.1f}%")
        if st.button("Restart"):
            st.session_state.update({'stage': 1, 'scores': [], 'spoken': False})
            st.rerun()

with tab2:
    st.header("Upload Handwriting Sample")
    file = st.file_uploader("Upload an image (JPG/PNG) of handwriting", type=['png', 'jpg', 'jpeg'])
    if file:
        img = Image.open(file).convert('L')
        gray_up = np.array(img)
        st.image(img, caption="Uploaded Sample", width=400)
        
        if st.button("Analyze Uploaded File"):
            
            score = run_prediction(gray_up, run_dl=True)
            label, color = get_severity(score)
            
            st.divider()
            if label == "Normal":
                st.success(f"### Detection: {label}")
            else:
                st.markdown(f"### Detection: :{color}[{label}]")
            st.progress(score)
            st.write(f"Certainty: {score*100:.1f}%")
