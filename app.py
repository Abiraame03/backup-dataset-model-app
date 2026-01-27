import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import os
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# --- I. Configuration ---
st.set_page_config(page_title="Dyslexia Severity Analyzer", layout="wide")

st.title("🧠 Precision Dyslexia Severity Analyzer")
st.markdown("---")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"
GLOBAL_THRESHOLD = 0.51  
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

# --- II. Audio Instruction Component ---
def speak_text(text):
    components.html(f"""
        <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.9;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- III. Logic & Accuracy Engine ---

def get_severity(prob):
    """Refined severity classification logic."""
    if prob < GLOBAL_THRESHOLD:
        return "Normal", "green", "✅"
    elif GLOBAL_THRESHOLD <= prob < 0.65:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.65 <= prob < 0.85:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def extract_features(img):
    """Optimized feature extraction for RF (64x64 HOG)."""
    img_res = cv2.resize(img, (64, 64))
    features = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    placeholders = [np.var(img_res), np.mean(img_res), 0, 0] 
    return np.concatenate([features, placeholders]).reshape(1, -1)



def predict_logic(gray_img, current_stage):
    """
    TASK-AWARE ROUTING:
    - Level 1 (Chars): RF is high weight (80%) because DL wasn't trained on chars.
    - Level 2 (Words): Hybrid weight (50/50).
    - Level 3 (Sentences): DL is high weight (80%) as it is a sentence specialist.
    """
    # 1. Preprocess & Denoise
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        gray_img = cv2.resize(gray_img[y:y+h, x:x+w], (160, 160))
    else:
        gray_img = cv2.resize(gray_img, (160, 160))

    rf_p = rf_m.predict_proba(extract_features(gray_img))[0][1] if rf_m else 0.0
    
    dl_p = 0.0
    if dl_m:
        rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_m.predict(inp, verbose=0)[0][0])

    # 2. Dynamic Weighting based on Task Type
    if current_stage == 1:
        # Heavily favor RF for characters
        final_score = (rf_p * 0.8) + (dl_p * 0.2)
    elif current_stage == 2:
        # Balanced for words
        final_score = (rf_p * 0.5) + (dl_p * 0.5)
    else:
        # Heavily favor DL for sentences
        final_score = (rf_p * 0.2) + (dl_p * 0.8)

    return final_score, rf_p, dl_p

# --- IV. Model Loading ---
@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except: pass
    return rf, dl

rf_m, dl_m = load_models()

# --- V. UI Workflow ---

if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'results': [], 'spoken': False, 'model_data': []})

t1, t2 = st.tabs(["✍️ Assessment Canvas", "📤 External File Analysis"])

with t1:
    if st.session_state.stage <= 3:
        age = st.sidebar.slider("Participant Age", 5, 12, 8)
        task = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"][st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_text(f"Level {st.session_state.stage}. {task}")
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        st.info(f"🔊 Task: {task}")
        
        canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=700, key=f"c{st.session_state.stage}")
        
        if st.button(f"Submit Level {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                score, r_p, d_p = predict_logic(gray, st.session_state.stage)
                
                st.session_state.results.append(score)
                st.session_state.model_data.append({"RF": r_p, "DL": d_p})
                st.session_state.stage += 1
                st.session_state.spoken = False 
                st.rerun()
    else:
        # OVERALL ASSESSMENT REPORT
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score)
        
        st.header(f"{icon} Final Assessment Report")
        
        if label == "Normal":
            st.balloons()
            st.success(f"### Final Result: {label}")
        else:
            st.error(f"### Final Result: {label}")

        

        # Detailed Breakdown Table
        with st.expander("🔍 View Technical Model Predictions"):
            st.write("Below is the raw data from each model across the three tasks:")
            breakdown = []
            for i, data in enumerate(st.session_state.model_data):
                breakdown.append({
                    "Level": i+1,
                    "RF Prediction (General)": f"{data['RF']*100:.1f}%",
                    "DL Prediction (Sentence)": f"{data['DL']*100:.1f}%",
                    "Weighted Score": f"{st.session_state.results[i]*100:.1f}%"
                })
            st.table(breakdown)

        st.divider()
        st.metric("Aggregate Risk Probability", f"{avg_score*100:.1f}%", delta=f"{avg_score - GLOBAL_THRESHOLD:.2f}", delta_color="inverse")
        
        if st.button("Start New Test"):
            st.session_state.update({'stage': 1, 'results': [], 'spoken': False, 'model_data': []})
            st.rerun()

with t2:
    st.header("Upload Handwriting Sample")
    up = st.file_uploader("Upload an image of written sentences", type=['png', 'jpg', 'jpeg'])
    if up:
        img = np.array(Image.open(up).convert('L'))
        st.image(up, width=400)
        if st.button("Analyze Sentences"):
            # For general uploads, we assume sentence-level analysis
            score, r_p, d_p = predict_logic(img, current_stage=3)
            label, color, icon = get_severity(score)
            
            st.markdown(f"## {icon} Detection: :{color}[{label}]")
            st.progress(score)
            st.write(f"RF Model Certainty: {r_p*100:.1f}%")
            st.write(f"DL Model Certainty: {d_p*100:.1f}%")
