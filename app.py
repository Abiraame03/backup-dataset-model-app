import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import os
import time
from datetime import datetime
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# --- I. Configuration ---
st.set_page_config(page_title="Precision Dyslexia Analyzer", layout="wide")
st.title("🧠 Coordination & Dyslexia Severity Analyzer")

# PATHS: Update these to your local paths
RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "models/mobilenetv2_bilstm_final.h5" 

IMG_SIZE_DL = (160, 160) 
CANVAS_THRESHOLD = 0.50

PUZZLES = {
    "Beginner (5-7)": {
        1: "Draw the letters b and d slowly.",
        2: "Write the word CAT in large letters.",
        3: "Write the sentence: The sun is hot."
    },
    "Advanced (8-12)": {
        1: "Draw the letters p, q, b, and d.",
        2: "Write the word MOUNTAIN clearly.",
        3: "Write: The quick brown fox jumps."
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
            st.error(f"DL Model Load Error: {e}")
    return rf, dl

rf_m, dl_m = load_models()

# --- III. Logic Engine ---
def get_severity(prob, threshold):
    if prob < threshold:
        return "Normal", "green", "✅"
    elif threshold <= prob < (threshold + 0.15):
        return "Mild Dyslexia", "blue", "⚠️"
    elif (threshold + 0.15) <= prob < (threshold + 0.30):
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def ensemble_predict(canvas_rgba, stage):
    # 1. Convert to Grayscale
    # Canvas comes in as RGBA (0-255)
    img_rgba = canvas_rgba.astype(np.uint8)
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
    
    # 2. RF Path (HOG Features)
    # We use a 64x64 version for the RF model
    img_64 = cv2.resize(gray, (64, 64))
    feats = hog(img_64, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    rf_inp = np.concatenate([feats, [np.var(img_64), np.mean(img_64), 0, 0]]).reshape(1, -1)
    rf_p = rf_m.predict_proba(rf_inp)[0][1] if rf_m else 0.0

    # 3. DL Path (The "Old Model" Integration)
    dl_p = 0.0
    if dl_m:
        # Convert grayscale to RGB (3 channels) as required by MobileNetV2
        dl_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Resize to exactly 160x160 (from your old app code)
        dl_img = cv2.resize(dl_img, IMG_SIZE_DL)
        # Normalize 1/255.0 (CRITICAL for the old model)
        dl_inp = np.expand_dims(dl_img / 255.0, axis=0)
        
        prediction = dl_m.predict(dl_inp, verbose=0)
        dl_p = float(prediction[0][0]) 

    # 4. Final Weighted Scoring
    if stage == 1: 
        score = (rf_p * 0.9 + dl_p * 0.1)
    elif stage == 2: 
        score = (rf_p * 0.5 + dl_p * 0.5)
    else: 
        score = (rf_p * 0.2 + dl_p * 0.8)
        
    return score, rf_p, dl_p

# --- IV. UI Flow ---
if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'results': [], 'rf_raw': [], 'dl_raw': [], 'start_time': time.time()})

age = st.sidebar.slider("Age", 5, 12, 7)
task_list = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"]

if st.session_state.stage <= 3:
    current_task = task_list[st.session_state.stage]
    st.subheader(f"Task {st.session_state.stage} of 3")
    st.info(f"📝 **Task:** {current_task}")

    canvas = st_canvas(
        stroke_width=5, stroke_color="#000", background_color="#FFF",
        height=350, width=800, key=f"canvas_{st.session_state.stage}"
    )

    if st.button("Submit Analysis", use_container_width=True):
        if canvas.image_data is not None:
            # Check if there is enough 'ink' on the canvas
            if np.sum(canvas.image_data[:, :, :3] < 255) > 500:
                with st.spinner("Processing drawing..."):
                    score, r_p, d_p = ensemble_predict(canvas.image_data, st.session_state.stage)
                    st.session_state.results.append(score)
                    st.session_state.rf_raw.append(r_p)
                    st.session_state.dl_raw.append(d_p)
                    st.session_state.stage += 1
                    st.rerun()
            else:
                st.warning("Please draw on the canvas before submitting.")
else:
    # --- Final Results View (Matches your screenshot) ---
    avg_score = np.mean(st.session_state.results)
    label, color, icon = get_severity(avg_score, CANVAS_THRESHOLD)
    test_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(f"### Final Result: {label} {icon}")
    st.write(f"🕒 **Test Date:** {test_date}")

    with st.expander("📊 Detailed Model Performance Breakdown", expanded=True):
        summary_data = []
        for i in range(3):
            summary_data.append({
                "Level": i+1,
                "RF Prediction": f"{st.session_state.rf_raw[i]*100:.1f}%",
                "DL (Old Model)": f"{st.session_state.dl_raw[i]*100:.1f}%",
                "Weighted Score": f"{st.session_state.results[i]*100:.1f}%"
            })
        st.table(summary_data)

    st.divider()
    st.metric("Aggregate Index", f"{avg_score*100:.1f}%")
    from unity_launcher import show_unity_button
    show_unity_button()
    if st.button("Restart Assessment"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
