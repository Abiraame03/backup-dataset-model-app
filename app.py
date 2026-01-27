import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import os
import time
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# --- I. Configuration ---
st.set_page_config(page_title="Dyslexia Severity Analyzer", layout="wide")

# Title and Header
st.title("🧠 Coordination & Dyslexia Severity Analyzer")
st.markdown("---")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"
GLOBAL_THRESHOLD = 0.51  # Set to 51% per request
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

# --- II. Audio Input Component ---
def speak_text(text):
    """Function to trigger browser Text-to-Speech"""
    components.html(f"""
        <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.9;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- III. Model Loading ---
@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except Exception as e:
            st.warning(f"DL Model could not load: {e}")
    return rf, dl

rf_model, dl_model = load_models()

# --- IV. Logic & Severity Engine ---

def get_severity(prob):
    """Calculates severity based on the 51% threshold references."""
    if prob < GLOBAL_THRESHOLD:
        return "Normal", "green", "✅"
    elif GLOBAL_THRESHOLD <= prob < 0.65:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.65 <= prob < 0.85:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def extract_features(img):
    img_res = cv2.resize(img, (64, 64))
    features = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    placeholders = [np.var(img_res), np.mean(img_res), 0, 0]
    return np.concatenate([features, placeholders]).reshape(1, -1)

def predict_all(gray_img, use_dl=True):
    rf_p, dl_p = 0.0, 0.0
    if rf_model:
        rf_p = rf_model.predict_proba(extract_features(gray_img))[0][1]
    if dl_model and use_dl:
        rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(cv2.resize(rgb, IMG_SIZE_DL)/255.0, axis=0)
        dl_p = float(dl_model.predict(inp, verbose=0)[0][0])
    
    # Returning raw probability (0.0 to 1.0)
    # Weighted ensemble for better accuracy (DL is 60% of score for Level 2 and 3)
    return (rf_p * 0.4 + dl_p * 0.6) if use_dl else rf_p

# --- V. UI Tabs ---

if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'results': [], 'spoken': False})

t1, t2 = st.tabs(["✍️ Assessment Canvas", "📤 Upload Sample"])

with t1:
    if st.session_state.stage <= 3:
        age = st.sidebar.slider("Age", 5, 12, 7)
        task = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"][st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_text(f"Task {st.session_state.stage}. {task}")
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        st.info(f"🔊 **Listening to Task:** {task}")
        
        canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=700, key=f"v{st.session_state.stage}")
        
        if st.button(f"Submit Task {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                # Level 2 and 3 use DL prediction alongside RF
                score = predict_all(gray, use_dl=(st.session_state.stage >= 2))
                st.session_state.results.append(score)
                st.session_state.stage += 1
                st.session_state.spoken = False 
                st.rerun()
    else:
        # OVERALL ASSESSMENT SECTION
        st.header("🏁 Overall Assessment Report")
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score)
        
        if label == "Normal":
            st.balloons()
            st.success(f"### Overall Result: {label} {icon}")
        else:
            st.error(f"### Overall Result: {label} {icon}")
        
        

        st.divider()
        st.write(f"**Average Analysis Probability:** {avg_score*100:.1f}%")
        st.write(f"Based on your performance across character, word, and sentence levels, the system has detected markers consistent with a **{label}** profile.")
        
        if st.button("Start New Assessment"):
            st.session_state.update({'stage': 1, 'results': [], 'spoken': False})
            st.rerun()

with t2:
    st.header("Upload Handwriting Image")
    up = st.file_uploader("Upload a clear image of a sentence", type=['png', 'jpg', 'jpeg'])
    if up:
        img = Image.open(up).convert('L')
        st.image(img, width=400, caption="Uploaded Handwriting")
        if st.button("Run Comprehensive Analysis"):
            # Upload uses both models for highest accuracy
            score = predict_all(np.array(img), use_dl=True)
            label, color, icon = get_severity(score)
            
            st.divider()
            st.markdown(f"## {icon} Detection: :{color}[{label}]")
            st.progress(score)
            st.write(f"Detection Certainty: **{score*100:.1f}%**")
            st.caption(f"Note: Detection based on a reference threshold of {GLOBAL_THRESHOLD*100}%.")
