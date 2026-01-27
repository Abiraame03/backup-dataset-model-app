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
THRESHOLD_PATH = "best_threshold.json"
IMG_SIZE_DL = (160, 160)

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
        3: "Write: The quick brown fox jumps over the lazy dog."
    }
}
TIME_BENCHMARKS = {5:65, 6:60, 7:55, 8:50, 9:45, 10:40, 11:35, 12:30}

@st.cache_resource
def load_all_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"DL Model Error: {e}")
            
    thresh = 0.51
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, "r") as f:
                thresh = json.load(f).get('threshold', 0.51)
        except: pass
    return rf, dl, thresh

rf_model, dl_model, dl_threshold = load_all_models()

# --- II. Feature Extraction & Prediction Logic ---

def enhance_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def extract_rf_features(img, img_size=64):
    img = enhance_image(img)
    img = cv2.resize(img, (img_size, img_size))
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

def run_dual_prediction(gray_img, run_dl=False):
    """
    Runs input through models. 
    run_dl: If True, the Neural Model (DL) will process the image. 
            If False, only the RF model runs.
    """
    preds = {"rf": {"label": "N/A", "conf": 0}, "dl": {"label": "Skipped", "conf": 0}}
    
    # 1. RF Model Prediction (Always runs)
    if rf_model:
        rf_feats = extract_rf_features(gray_img).reshape(1, -1)
        prob_rf = rf_model.predict_proba(rf_feats)[0][1] * 100
        preds["rf"]["label"] = "Dyslexic" if prob_rf > 54.0 else "Normal"
        preds["rf"]["conf"] = prob_rf
        
    # 2. DL Model Prediction (Conditional for Level 2/Sentences)
    if dl_model and run_dl:
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        dl_input = cv2.resize(rgb_img, IMG_SIZE_DL) / 255.0
        dl_input = np.expand_dims(dl_input, axis=0)
        prob_dl_raw = float(dl_model.predict(dl_input, verbose=0)[0][0])
        preds["dl"]["label"] = "Dyslexic" if prob_dl_raw >= dl_threshold else "Normal"
        preds["dl"]["conf"] = prob_dl_raw * 100
        
    return preds

# --- III. Interface and Session Handling ---

def speak_task(text):
    components.html(f"<script>var msg = new SpeechSynthesisUtterance('{text}'); msg.rate = 0.85; window.speechSynthesis.speak(msg);</script>", height=0)

if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 
        'data': {"imgs": [], "times": [], "level_results": []}, 
        'start_time': None, 'spoken': False
    })

st.title("🧠 Coordination & Dual-Model Dyslexia Analyzer")
st.info(f"**Goal:** {GENERALIZED_GOAL}")

tab1, tab2 = st.tabs(["✍️ Writing Canvas", "📤 Upload Image"])

with st.sidebar:
    u_age = st.slider("Select Age", 5, 12, 7)
    if st.button("Reset Assessment"):
        st.session_state.update({'stage': 1, 'data': {"imgs": [], "times": [], "level_results": []}, 'start_time': None, 'spoken': False})
        st.rerun()

# --- TAB 1: CANVAS MODE ---
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
            if canvas_result.image_data is not None and st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                gray = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                # LOGIC CHANGE: Run DL only if stage is 2
                should_run_dl = (st.session_state.stage == 2)
                results = run_dual_prediction(gray, run_dl=should_run_dl)
                
                st.session_state.data["level_results"].append({
                    "rf_label": results["rf"]["label"],
                    "rf_conf": results["rf"]["conf"],
                    "dl_label": results["dl"]["label"],
                    "dl_conf": results["dl"]["conf"],
                    "time": elapsed,
                    "dl_active": should_run_dl # Store whether DL was used
                })
                st.session_state.stage += 1
                st.session_state.start_time = None
                st.session_state.spoken = False
                st.rerun()

    if len(st.session_state.data["level_results"]) > 0:
        st.divider()
        st.subheader("📊 Individual Level Model Outputs")
        cols = st.columns(3)
        for i, res in enumerate(st.session_state.data["level_results"]):
            with cols[i]:
                st.markdown(f"**Level {i+1}**")
                st.write(f"RF Model: **{res['rf_label']}** ({res['rf_conf']:.1f}%)")
                if res["dl_active"]:
                    st.write(f"DL Model: **{res['dl_label']}** ({res['dl_conf']:.1f}%)")
                else:
                    st.write("DL Model: *Not used for this level*")

# --- TAB 2: UPLOAD MODE ---
with tab2:
    st.markdown("### Sentence Analysis (Level 2 Deep Learning)")
    up = st.file_uploader("Upload Image for Sentence Analysis", type=['png', 'jpg', 'jpeg'])
    if up:
        img_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        gray_up = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(gray_up, width=300, caption="Uploaded Sentence Sample")
        
        if st.button("Run Sentence-Specific Analysis"):
            # For Upload tab, we assume this is the Sentence Level (Level 2)
            results = run_dual_prediction(gray_up, run_dl=True)
            c1, c2 = st.columns(2)
            with c1:
                st.write("### RF (Geometric)")
                st.markdown(f"**Result: {results['rf']['label']}**")
            with c2:
                st.write("### DL (Neural Sentence Model)")
                st.markdown(f"**Result: {results['dl']['label']}**")
                st.write(f"Confidence: {results['dl']['conf']:.2f}%")
