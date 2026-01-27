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

# --- II. Audio Instruction ---
def speak_text(text):
    components.html(f"<script>window.speechSynthesis.cancel(); var msg = new SynthesisUtterance('{text}'); window.speechSynthesis.speak(msg);</script>", height=0)

# --- III. Logic & Accuracy Engine (REFINED) ---

def preprocess_handwriting(gray_img):
    """
    Accuracy Fix: Crops the image to the actual handwriting.
    This prevents 'Normal' writing from being detected as 'Mild' 
    due to excessive white space or padding.
    """
    # Threshold to find the ink (assuming black ink on white background)
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = gray_img[y:y+h, x:x+w]
        return cv2.resize(cropped, (IMG_SIZE_DL[0], IMG_SIZE_DL[1]))
    return cv2.resize(gray_img, (IMG_SIZE_DL[0], IMG_SIZE_DL[1]))

def get_severity(prob):
    """
    Threshold Calibration:
    Added a slight buffer (0.04) to the 51% threshold to prevent 
    false 'Mild' positives for 'Borderline Normal' handwriting.
    """
    # We only classify as Dyslexic if probability significantly clears the baseline
    ADJUSTED_BASE = GLOBAL_THRESHOLD + 0.04 

    if prob < ADJUSTED_BASE:
        return "Normal", "green", "✅"
    elif ADJUSTED_BASE <= prob < 0.65:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.65 <= prob < 0.85:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def extract_features(img):
    img_res = cv2.resize(img, (64, 64))
    # HOG features help identify the 'jitter' in dyslexic strokes
    features = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    placeholders = [np.var(img_res), np.mean(img_res), 0, 0] 
    return np.concatenate([features, placeholders]).reshape(1, -1)



def predict_all(gray_img, rf_model, dl_model, use_dl=True):
    # Preprocess to focus only on the ink
    processed_img = preprocess_handwriting(gray_img)
    
    rf_p, dl_p = 0.0, 0.0
    
    if rf_model:
        feats = extract_features(processed_img)
        rf_p = rf_model.predict_proba(feats)[0][1]
    
    if dl_model and use_dl:
        rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_model.predict(inp, verbose=0)[0][0])
    
    # Accuracy Logic: For 'Normal' detections, the models usually disagree.
    # We use a 'Consensus' check to lower the probability if one model is very low.
    if use_dl:
        combined = (rf_p * 0.4 + dl_p * 0.6)
        # If one model says it's definitely normal (< 0.3), pull the average down
        if rf_p < 0.3 or dl_p < 0.3:
            combined *= 0.85 
        return combined
    return rf_p

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

# --- V. UI Integration ---

if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'results': [], 'spoken': False})

t1, t2 = st.tabs(["✍️ Canvas Assessment", "📤 File Analysis"])

with t1:
    if st.session_state.stage <= 3:
        age = st.sidebar.slider("Age", 5, 12, 7)
        task = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"][st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_text(task)
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=700, key=f"v{st.session_state.stage}")
        
        if st.button(f"Submit Task {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                score = predict_all(gray, rf_m, dl_m, use_dl=(st.session_state.stage >= 2))
                st.session_state.results.append(score)
                st.session_state.stage += 1
                st.session_state.spoken = False 
                st.rerun()
    else:
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score)
        
        

        if label == "Normal":
            st.balloons()
            st.success(f"### Final Assessment: {label} {icon}")
        else:
            st.error(f"### Final Assessment: {label} {icon}")
        
        st.metric("Risk Probability", f"{avg_score*100:.1f}%", delta_color="inverse")
        if st.button("Restart"):
            st.session_state.update({'stage': 1, 'results': [], 'spoken': False})
            st.rerun()

with t2:
    up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if up:
        img = np.array(Image.open(up).convert('L'))
        st.image(up, width=300)
        if st.button("Analyze File"):
            score = predict_all(img, rf_m, dl_m, use_dl=True)
            label, color, icon = get_severity(score)
            st.markdown(f"## {icon} Result: :{color}[{label}]")
            st.write(f"Certainty: {score*100:.1f}%")
