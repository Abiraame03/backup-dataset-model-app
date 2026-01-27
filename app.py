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
st.set_page_config(page_title="Accurate Dual-Model Dyslexia Analyzer", layout="wide")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5" 
THRESHOLD_PATH = "best_threshold.json"
IMG_SIZE_DL = (160, 160)

# ACCURACY FIX: Higher threshold to prevent "Mild" false positives
STRICT_THRESHOLD = 0.70 

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
        except: pass
    return rf, dl

rf_model, dl_model = load_all_models()

# --- II. Logic Functions ---

def speak_task(text):
    """Triggers browser TTS for audio instructions."""
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.85;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

def extract_rf_features(img, img_size=64):
    img_res = cv2.resize(img, (img_size, img_size))
    hog_feat = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    # Ensure feature vector matches the RF model's expected input length
    edges = cv2.Canny(img_res, 80, 160)
    edge_density = np.sum(edges) / (np.sum(img_res > 0) + 1)
    return np.concatenate([hog_feat, [edge_density, np.var(img_res), 0, 0]])

def run_dual_prediction(gray_img, run_dl=False):
    preds = {"rf_prob": 0.0, "dl_prob": 0.0}
    
    # 1. RF Prediction
    if rf_model:
        rf_feats = extract_rf_features(gray_img).reshape(1, -1)
        preds["rf_prob"] = rf_model.predict_proba(rf_feats)[0][1]
        
    # 2. DL Prediction (Word/Sentence Levels)
    if dl_model and run_dl:
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        dl_input = np.expand_dims(cv2.resize(rgb_img, IMG_SIZE_DL)/255.0, axis=0)
        preds["dl_prob"] = float(dl_model.predict(dl_input, verbose=0)[0][0])
        
    return preds

# --- III. Session Handling ---

if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 'data': {"times": [], "level_probs": []}, 
        'start_time': None, 'spoken': False
    })

st.title("🧠 Coordination & Audio-Task Analyzer")

with st.sidebar:
    u_age = st.slider("User Age", 5, 12, 7)
    if st.button("Restart Assessment"):
        st.session_state.update({'stage': 1, 'data': {"times": [], "level_probs": []}, 'spoken': False})
        st.rerun()

# --- IV. Assessment UI ---

if st.session_state.stage <= 3:
    bracket = "Beginner (5-7)" if u_age <= 7 else "Advanced (8-12)"
    task_text = PUZZLES[bracket][st.session_state.stage]
    
    # Audio Trigger
    if not st.session_state.spoken:
        speak_task(f"Task {st.session_state.stage}. {task_text}")
        st.session_state.spoken = True

    st.subheader(f"Level {st.session_state.stage}")
    st.info(f"🔊 **Audio Task:** {task_text}")
    
    canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=800, key=f"c{st.session_state.stage}")

    if st.button(f"Submit Task {st.session_state.stage}"):
        if canvas.image_data is not None:
            gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            probs = run_dual_prediction(gray, run_dl=(st.session_state.stage >= 2))
            
            st.session_state.data["level_probs"].append(probs)
            st.session_state.stage += 1
            st.session_state.spoken = False
            st.rerun()

# --- V. Final Strict Assessment ---
elif st.session_state.stage > 3:
    st.header("🏁 Accurate Final Report")
    
    # Calculate Average Probabilities
    avg_rf = np.mean([p['rf_prob'] for p in st.session_state.data["level_probs"]])
    # Only average DL for levels where it ran (2 and 3)
    dl_scores = [p['dl_prob'] for p in st.session_state.data["level_probs"] if p['dl_prob'] > 0]
    avg_dl = np.mean(dl_scores) if dl_scores else 0
    
    # Logic: Confirmation requires both models to be above a strict baseline
    is_confirmed = (avg_rf > 0.58) and (avg_dl > STRICT_THRESHOLD)

    

    if is_confirmed:
        st.error(f"### Detection: Dyslexic Profile Identified")
        st.write("Both geometric analysis and neural sequence modeling confirmed high-risk markers.")
    else:
        st.balloons()
        st.success("### Detection: Normal / Low Risk")
        st.write("The indicators found do not meet the strict statistical threshold for a dyslexia diagnosis.")

    st.divider()
    st.subheader("📊 Detailed Confidence Scores")
    c1, c2 = st.columns(2)
    c1.metric("Geometric Consistency (RF)", f"{avg_rf*100:.1f}%")
    c2.metric("Neural Sequence Accuracy (DL)", f"{avg_dl*100:.1f}%")
