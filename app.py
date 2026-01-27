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

# --- New Imports for the Second Model ---
import tensorflow as tf

# --- App Configuration ---
st.set_page_config(page_title="Dyslexia Detection & Severity Analyzer", layout="wide")

GENERALIZED_GOAL = "Goal: Improving Hand-Eye Coordination through steady strokes and consistent letter sizing."

# Auditory Task Content
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

# --- Model Loading ---
@st.cache_resource
def load_models():
    # Model 1: Random Forest
    rf_path = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    
    # Model 2: MobileNetV2-BiLSTM
    h5_path = "models/mobilenetv2_bilstm_final.h5"
    dl_model = tf.keras.models.load_model(h5_path) if os.path.exists(h5_path) else None
    
    # Load DL Metadata
    class_map = None
    if os.path.exists("models/class_indices_best.pkl"):
        with open("models/class_indices_best.pkl", "rb") as f:
            class_map = pickle.load(f)
            
    best_thresh = 0.5
    if os.path.exists("models/best_threshold.json"):
        with open("models/best_threshold.json", "r") as f:
            best_thresh = json.load(f).get('threshold', 0.5)

    return rf_model, dl_model, class_map, best_thresh

rf_model, dl_model, class_indices, dl_threshold = load_models()

# --- Session State ---
if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 
        'data': {"imgs": [], "times": [], "level_results": []}, 
        'start_time': None, 'spoken': False
    })

# --- Main UI ---
st.title("🧠 Coordination & Auditory Dyslexia Analyzer")
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
            if canvas_result.image_data is not None:
                elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                gray = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                # 1. RF Prediction (54% Threshold)
                rf_feats = extract_hog_features(gray).reshape(1, -1)
                rf_prob = rf_model.predict_proba(rf_feats)[0][1] * 100
                rf_label = "Dyslexic" if rf_prob > 54.0 else "Normal"
                
                # 2. MobileNet-BiLSTM Prediction
                dl_img = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), (224, 224)) / 255.0
                dl_img = np.expand_dims(dl_img, axis=0)
                dl_prob_raw = dl_model.predict(dl_img, verbose=0)[0][0]
                dl_label = "Dyslexic" if dl_prob_raw > dl_threshold else "Normal"
                
                st.session_state.data["imgs"].append(gray)
                st.session_state.data["times"].append(elapsed)
                st.session_state.data["level_results"].append({
                    "rf": {"label": rf_label, "conf": rf_prob},
                    "dl": {"label": dl_label, "conf": dl_prob_raw * 100}
                })
                
                st.session_state.stage += 1
                st.session_state.start_time = None
                st.session_state.spoken = False
                st.rerun()

    # --- Live Multi-Model Results ---
    if len(st.session_state.data["level_results"]) > 0:
        st.divider()
        st.subheader("📊 Individual Level Model Outputs")
        cols = st.columns(3)
        for i, res in enumerate(st.session_state.data["level_results"]):
            with cols[i]:
                st.markdown(f"**Level {i+1} Analysis**")
                st.write(f"RF Model: **{res['rf']['label']}** ({res['rf']['conf']:.1f}%)")
                st.write(f"Deep Model: **{res['dl']['label']}** ({res['dl']['conf']:.1f}%)")

    # --- Final Overall Report ---
    if st.session_state.stage > 3:
        st.divider()
        st.header("🏁 Overall Analysis Report")
        
        # Majority voting between both models across 3 stages
        rf_dyslexic = sum(1 for r in st.session_state.data["level_results"] if r["rf"]["label"] == "Dyslexic")
        dl_dyslexic = sum(1 for r in st.session_state.data["level_results"] if r["dl"]["label"] == "Dyslexic")
        
        overall_is_dyslexic = (rf_dyslexic + dl_dyslexic) >= 3 # Consensus check

        if not overall_is_dyslexic:
            st.balloons()
            st.success("### Overall Detection: Normal / Non-Dyslexic")
            st.write("Both models indicate standard handwriting development.")
        else:
            avg_time = sum(st.session_state.data["times"]) / 3
            target = TIME_BENCHMARKS.get(u_age, 30)
            diff = avg_time - target
            
            if diff < 5: severity, color = "Mild Risk", "blue"
            elif 5 <= diff < 15: severity, color = "Moderate Risk", "orange"
            else: severity, color = "Severe Risk", "red"

            

            st.error(f"### Overall Detection: Dyslexic Profile Confirmed")
            st.markdown(f"## Final Severity: :{color}[{severity}]")
            st.write(f"**Detailed Summary:** Consensus reached between HOG-RF and MobileNet-BiLSTM models. Performance at Age {u_age} shows a {round(diff, 1)}s delay relative to peer benchmarks.")

# --- TAB 2: UPLOAD MODE ---
with tab2:
    up = st.file_uploader("Upload Image", type=['png', 'jpg'])
    if up and rf_model and dl_model:
        file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(img_gray, width=300)
        
        if st.button("Predict Upload"):
            # RF
            u_rf_feats = extract_hog_features(img_gray).reshape(1, -1)
            u_rf_prob = rf_model.predict_proba(u_rf_feats)[0][1] * 100
            
            # DL
            u_dl_img = cv2.resize(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB), (224, 224)) / 255.0
            u_dl_img = np.expand_dims(u_dl_img, axis=0)
            u_dl_prob = dl_model.predict(u_dl_img, verbose=0)[0][0] * 100
            
            st.write(f"RF Model Confidence: {u_rf_prob:.1f}%")
            st.write(f"Deep Model Confidence: {u_dl_prob:.1f}%")
            
            final_res = "Dyslexic" if (u_rf_prob > 51.0 or u_dl_prob > (dl_threshold*100)) else "Normal"
            st.subheader(f"Final Decision: {final_res}")
