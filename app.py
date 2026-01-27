import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import json
import time
import os
import random
import streamlit.components.v1 as components
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas

# --- I. Configuration and Model Loading ---
st.set_page_config(page_title="Dual-Model Dyslexia Analyzer", layout="wide")

# File paths - Looking in the root directory
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
def load_all_resources():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    is_sim = True
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
            is_sim = False
        except Exception as e:
            st.error(f"DL Model Load Error: {e}")
    
    thresh = 0.51
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, "r") as f:
                thresh = json.load(f).get('threshold', 0.51)
        except: pass
    return rf, dl, thresh, is_sim

rf_model, dl_model, dl_threshold, IS_SIMULATION_MODE = load_all_resources()

# --- II. Helper Functions ---

def speak_task(text):
    components.html(f"<script>var msg = new SpeechSynthesisUtterance('{text}'); msg.rate = 0.85; window.speechSynthesis.speak(msg);</script>", height=0)

def get_severity_label(prob):
    if prob <= 0.10: return "Low Risk"
    elif prob <= 0.30: return "Mild Risk"
    elif prob <= 0.70: return "Moderate Risk"
    else: return "Severe Risk"

def extract_rf_features(img):
    img_resized = cv2.resize(img, (64, 64))
    hog_feat = hog(img_resized, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    edges = cv2.Canny(img_resized, 80, 160)
    edge_density = np.sum(edges) / (np.sum(img_resized > 0) + 1)
    # Pad to match original RF input shape if necessary
    return np.concatenate([hog_feat, [edge_density, np.var(img_resized), 0, 0]])

def run_dual_prediction(gray_img):
    res = {"rf": {}, "dl": {}}
    
    # RF Logic
    if rf_model:
        rf_feats = extract_rf_features(gray_img).reshape(1, -1)
        prob_rf = rf_model.predict_proba(rf_feats)[0][1]
        res["rf"]["label"] = "Dyslexic" if prob_rf > 0.51 else "Normal"
        res["rf"]["conf"] = prob_rf * 100

    # DL Logic with Simulation Fallback
    prob_dl = 0.0
    if not IS_SIMULATION_MODE:
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        dl_input = np.expand_dims(cv2.resize(rgb_img, IMG_SIZE_DL)/255.0, axis=0)
        prob_dl = float(dl_model.predict(dl_input, verbose=0)[0][0])
    else:
        prob_dl = random.uniform(0.51, 0.95) if random.choice([True, False]) else random.uniform(0.05, 0.49)

    res["dl"]["prob"] = prob_dl
    res["dl"]["label"] = "Dyslexic" if prob_dl >= dl_threshold else "Non-dyslexic"
    res["dl"]["severity"] = get_severity_label(prob_dl)
    res["dl"]["conf"] = prob_dl * 100 if res["dl"]["label"] == "Dyslexic" else (1.0 - prob_dl) * 100
    return res

# --- III. Session State & UI ---

if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 
        'data': {"imgs": [], "times": [], "level_results": []}, 
        'start_time': None, 'spoken': False
    })

with st.sidebar:
    st.title("Settings")
    u_age = st.slider("Select Age", 5, 12, 7)
    st.divider()
    if st.button("Reset Assessment"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()
    if IS_SIMULATION_MODE:
        st.warning("Running DL in Simulation Mode")

tab1, tab2 = st.tabs(["✍️ Interactive Assessment", "📤 Quick Upload"])

with tab1:
    if st.session_state.stage <= 3:
        bracket = "Beginner (5-7)" if u_age <= 7 else "Advanced (8-12)"
        task_text = PUZZLES[bracket][st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_task(f"Level {st.session_state.stage}. {task_text}")
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}: {task_text}")
        canvas_res = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=800, key=f"c{st.session_state.stage}")

        if canvas_res.json_data and len(canvas_res.json_data["objects"]) > 0:
            if st.session_state.start_time is None: st.session_state.start_time = time.time()

        if st.button(f"Submit Level {st.session_state.stage}"):
            if canvas_res.image_data is not None and st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                gray = cv2.cvtColor(canvas_res.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                preds = run_dual_prediction(gray)
                st.session_state.data["times"].append(elapsed)
                st.session_state.data["level_results"].append(preds)
                
                st.session_state.stage += 1
                st.session_state.start_time = None
                st.session_state.spoken = False
                st.rerun()

    # --- Final Report Logic ---
    if st.session_state.stage > 3:
        st.header("🏁 Final Analysis Report")
        
        # Aggregate logic
        total_dl_prob = sum([r["dl"]["prob"] for r in st.session_state.data["level_results"]]) / 3
        final_label = "Dyslexic" if total_dl_prob >= dl_threshold else "Non-dyslexic"
        final_sev = get_severity_label(total_dl_prob)
        
        if final_label == "Dyslexic":
            st.error(f"## 🔴 DYSLEXIA PROFILE DETECTED")
            st.metric("Aggregate Risk", f"{total_dl_prob*100:.1f}%", final_sev)
        else:
            st.success(f"## 🟢 NORMAL HANDWRITING PROFILE")
            st.balloons()

        # Detailed breakdown per task
        cols = st.columns(3)
        for i, res in enumerate(st.session_state.data["level_results"]):
            with cols[i]:
                st.markdown(f"**Task {i+1} Analysis**")
                st.write(f"RF: {res['rf']['label']} ({res['rf']['conf']:.1f}%)")
                st.write(f"DL: {res['dl']['label']} ({res['dl']['conf']:.1f}%)")
                st.caption(f"Category: {res['dl']['severity']}")

with tab2:
    up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if up:
        img_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        gray_up = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(gray_up, width=400, caption="Uploaded Sample")
        
        if st.button("Run Full Dual Analysis"):
            res = run_dual_prediction(gray_up)
            c1, c2 = st.columns(2)
            with c1:
                st.info("### Random Forest")
                st.write(f"Label: **{res['rf']['label']}**")
                st.progress(res['rf']['conf']/100)
            with c2:
                st.info("### Deep Learning")
                st.write(f"Label: **{res['dl']['label']}**")
                st.write(f"Severity: **{res['dl']['severity']}**")
                st.progress(res['dl']['prob'])
