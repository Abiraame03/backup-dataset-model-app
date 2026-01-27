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
# Forced threshold as requested
GLOBAL_THRESHOLD = 0.51 
IMG_SIZE_DL = (160, 160)

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

# --- II. Severity Logic Helper ---

def get_severity_label(probability):
    """
    Categorizes the probability based on the 51% baseline.
    """
    prob_val = probability / 100.0 if probability > 1 else probability
    
    if prob_val < GLOBAL_THRESHOLD:
        return "Normal", "green"
    elif GLOBAL_THRESHOLD <= prob_val < 0.65:
        return "Mild Dyslexia", "blue"
    elif 0.65 <= prob_val < 0.85:
        return "Moderate Dyslexia", "orange"
    else:
        return "Severe Dyslexia", "red"

# --- III. Prediction Logic ---

def extract_rf_features(img, img_size=64):
    img_res = cv2.resize(img, (img_size, img_size))
    hog_feat = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    edges = cv2.Canny(img_res, 80, 160)
    edge_density = np.sum(edges) / (np.sum(img_res > 0) + 1)
    return np.concatenate([hog_feat, [edge_density, np.var(img_res), 0, 0]])

def run_dual_prediction(gray_img, run_dl=False):
    # We use raw probabilities to determine severity later
    results = {"rf_prob": 0.0, "dl_prob": 0.0}
    
    if rf_model:
        rf_feats = extract_rf_features(gray_img).reshape(1, -1)
        results["rf_prob"] = rf_model.predict_proba(rf_feats)[0][1]
        
    if dl_model and run_dl:
        rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        dl_input = np.expand_dims(cv2.resize(rgb_img, IMG_SIZE_DL)/255.0, axis=0)
        results["dl_prob"] = float(dl_model.predict(dl_input, verbose=0)[0][0])
        
    return results

# --- IV. Interface ---

if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 'data': {"level_results": []}, 
        'spoken': False
    })

st.title("🧠 Coordination & Severity-Based Dyslexia Analyzer")

tab1, tab2 = st.tabs(["✍️ Writing Canvas", "📤 Upload Image"])

with tab1:
    if st.session_state.stage <= 3:
        u_age = st.sidebar.slider("Select Age", 5, 12, 7)
        bracket = "Beginner (5-7)" if u_age <= 7 else "Advanced (8-12)"
        task_text = PUZZLES[bracket][st.session_state.stage]
        
        st.subheader(f"Level {st.session_state.stage}: {task_text}")
        canvas_result = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=800, key=f"c{st.session_state.stage}")

        if st.button(f"Submit Level {st.session_state.stage}"):
            if canvas_result.image_data is not None:
                gray = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                probs = run_dual_prediction(gray, run_dl=(st.session_state.stage >= 2))
                
                st.session_state.data["level_results"].append(probs)
                st.session_state.stage += 1
                st.rerun()

    if st.session_state.stage > 3:
        st.header("🏁 Final Severity Assessment")
        
        # Calculate Average Probability across all tasks
        # We average RF for all and DL for levels 2/3
        all_probs = []
        for i, res in enumerate(st.session_state.data["level_results"]):
            all_probs.append(res["rf_prob"])
            if res["dl_prob"] > 0: all_probs.append(res["dl_prob"])
        
        avg_probability = np.mean(all_probs)
        label, color = get_severity_label(avg_probability)
        
        

        if label == "Normal":
            st.balloons()
            st.success(f"### Result: {label}")
        else:
            st.markdown(f"### Result: :{color}[{label}]")
        
        st.write(f"Confidence Score: **{avg_probability*100:.1f}%** (Threshold: {GLOBAL_THRESHOLD*100}%)")
        
        st.divider()
        if st.button("Restart Assessment"):
            st.session_state.update({'stage': 1, 'data': {"level_results": []}})
            st.rerun()

with tab2:
    st.header("Single Sample Upload")
    up = st.file_uploader("Choose handwriting image...", type=['png', 'jpg', 'jpeg'])
    if up:
        img = Image.open(up).convert('L')
        gray_up = np.array(img)
        st.image(up, width=400, caption="Uploaded Sample")
        
        if st.button("Run Severity Analysis"):
            res = run_dual_prediction(gray_up, run_dl=True)
            # Use max probability for a single upload to be cautious
            max_prob = max(res["rf_prob"], res["dl_prob"])
            label, color = get_severity_label(max_prob)
            
            st.subheader("Analysis Breakdown")
            st.markdown(f"**Overall Classification: :{color}[{label}]**")
            st.write(f"RF Confidence: {res['rf_prob']*100:.1f}%")
            st.write(f"DL Confidence: {res['dl_prob']*100:.1f}%")
