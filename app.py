import streamlit as st
import numpy as np
import cv2
import joblib
import time
import os
import pickle
import json
import streamlit.components.v1 as components
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- Deep Learning Import ---
import tensorflow as tf

# --- App Configuration ---
st.set_page_config(page_title="Dyslexia Detection & Severity Analyzer", layout="wide")

GENERALIZED_GOAL = "Goal: Improving Hand-Eye Coordination through steady strokes and consistent letter sizing."
PUZZLES = {
    "Beginner (5-7)": {
        1: "Draw the letters b and d slowly and clearly.",
        2: "Write the word CAT using large capital letters.",
        3: "Write the sentence: The sun is hot."
    },
    "Advanced (8-12)": {
        1: "Draw the letters p, q, b, and d in a single row.",
        2: "Write the word MOUNTAIN in your best handwriting.",
        3: "Write: The quick brown fox."
    }
}
TIME_BENCHMARKS = {5:65, 6:60, 7:55, 8:50, 9:45, 10:40, 11:35, 12:30}

# --- Auditory Helper ---
def speak_task(text):
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 0.85; 
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- Feature Extraction (RF Model) ---
def enhance_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def extract_hog_features(img, img_size=64):
    img = enhance_image(cv2.resize(img, (img_size, img_size)))
    hog_feat = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    edges = cv2.Canny(img, 80, 160)
    edge_density = np.sum(edges) / (np.sum(img > 0) + 1)
    intensity_var = np.var(img)
    horiz_proj = np.sum(img, axis=1)
    spacing_var = np.var(horiz_proj)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    stroke_profile = np.sum(binary > 0, axis=1)
    stroke_width_var = np.var(stroke_profile)
    return np.concatenate([hog_feat, [edge_density, intensity_var, spacing_var, stroke_width_var]])

# --- Model Loading (Updated Paths to match your GitHub) ---
@st.cache_resource
def load_all_models():
    # 1. RF Model
    rf_path = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    
    # 2. Deep Learning Model (Removed 'models/' prefix)
    h5_path = "mobilenetv2_bilstm_final .h5" 
    dl_model = tf.keras.models.load_model(h5_path) if os.path.exists(h5_path) else None
    
    # 3. DL Metadata
    dl_thresh = 0.51
    if os.path.exists("best_threshold.json"):
        with open("best_threshold.json", "r") as f:
            dl_thresh = json.load(f).get('threshold', 0.51)
            
    return rf, dl_model, dl_thresh

rf_model, dl_model, dl_threshold = load_all_models()

# --- Session State ---
if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 
        'data': {"imgs": [], "times": [], "level_results": []}, 
        'start_time': None, 'spoken': False
    })

st.title("🧠 Coordination & Dual-Model Dyslexia Analyzer")
st.info(f"**Goal:** {GENERALIZED_GOAL}")

tab1, tab2 = st.tabs(["✍️ Writing Canvas", "📤 Upload Analysis"])

with st.sidebar:
    u_age = st.slider("Select Age", 5, 12, 7)
    if st.button("Reset Assessment"):
        st.session_state.update({'stage': 1, 'data': {"imgs": [], "times": [], "level_results": []}, 'start_time': None, 'spoken': False})
        st.rerun()

# --- TAB 1: CANVAS ---
with tab1:
    if st.session_state.stage <= 3:
        bracket = "Beginner (5-7)" if u_age <= 7 else "Advanced (8-12)"
        task_text = PUZZLES[bracket][st.session_state.stage]
        if not st.session_state.spoken:
            speak_task(f"Level {st.session_state.stage}. {task_text}")
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        canvas_result = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=800, key=f"c{st.session_state.stage}")

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()

        if st.button(f"Submit Level {st.session_state.stage}"):
            if canvas_result.image_data is not None:
                elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                gray = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                # --- RF Prediction (54% Threshold) ---
                rf_prob = rf_model.predict_proba(extract_hog_features(gray).reshape(1, -1))[0][1] * 100 if rf_model else 0
                rf_label = "Dyslexic" if rf_prob > 54.0 else "Normal"
                
                # --- DL Prediction ---
                dl_label, dl_prob = "N/A", 0
                if dl_model:
                    dl_img = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), (160, 160)) / 255.0
                    dl_img = np.expand_dims(dl_img, axis=0)
                    dl_prob_raw = dl_model.predict(dl_img, verbose=0)[0][0]
                    dl_prob = dl_prob_raw * 100
                    dl_label = "Dyslexic" if dl_prob_raw >= dl_threshold else "Normal"

                st.session_state.data["level_results"].append({
                    "rf": {"label": rf_label, "conf": rf_prob},
                    "dl": {"label": dl_label, "conf": dl_prob},
                    "time": elapsed
                })
                st.session_state.data["times"].append(elapsed)
                st.session_state.stage += 1
                st.session_state.start_time = None
                st.session_state.spoken = False
                st.rerun()

    # Results Display
    if len(st.session_state.data["level_results"]) > 0:
        st.divider()
        st.subheader("📊 Individual Level Model Outputs")
        cols = st.columns(3)
        for i, res in enumerate(st.session_state.data["level_results"]):
            with cols[i]:
                st.markdown(f"**Level {i+1}**")
                st.write(f"RF Model: **{res['rf']['label']}** ({res['rf']['conf']:.1f}%)")
                st.write(f"Deep Model: **{res['dl']['label']}** ({res['dl']['conf']:.1f}%)")

    # Final Report
    if st.session_state.stage > 3:
        st.divider()
        st.header("🏁 Overall Analysis Report")
        
        dys_count = sum(1 for r in st.session_state.data["level_results"] if r['rf']['label'] == "Dyslexic" or r['dl']['label'] == "Dyslexic")
        
        if dys_count < 2:
            st.balloons()
            st.success("### Overall Detection: Normal / Non-Dyslexic")
        else:
            avg_time = sum(st.session_state.data["times"]) / 3
            target = TIME_BENCHMARKS.get(u_age, 30)
            diff = avg_time - target
            severity, color = ("Severe Risk", "red") if diff > 15 else (("Moderate Risk", "orange") if diff > 5 else ("Mild Risk", "blue"))
            
            st.error(f"### Overall Detection: Dyslexic Profile Confirmed")
            st.markdown(f"## Final Severity: :{color}[{severity}]")
            st.write(f"**Age:** {u_age} | **Avg Time:** {avg_time:.1f}s | **Delay:** {diff:.1f}s")

# --- TAB 2: UPLOAD ---
with tab2:
    up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if up:
        img_pil = Image.open(up).convert("RGB")
        st.image(img_pil, width=300)
        if st.button("Run Prediction"):
            img_np = np.array(img_pil)
            gray_up = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # RF
            rf_u = rf_model.predict_proba(extract_hog_features(gray_up).reshape(1, -1))[0][1] * 100
            # DL
            dl_u_img = np.expand_dims(cv2.resize(img_np, (160, 160)) / 255.0, axis=0)
            dl_u_prob = dl_model.predict(dl_u_img, verbose=0)[0][0] * 100
            
            st.write(f"RF Confidence: {rf_u:.1f}%")
            st.write(f"Deep Model Confidence: {dl_u_prob:.1f}%")
