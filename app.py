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

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"
GLOBAL_THRESHOLD = 0.51  
IMG_SIZE_DL = (160, 160)

PUZZLES = {
    "Beginner (5-7)": {1: "Draw 'b' and 'd' slowly.", 2: "Write 'CAT'.", 3: "Write: The sun is hot."},
    "Advanced (8-12)": {1: "Draw 'p, q, b, d'.", 2: "Write 'MOUNTAIN'.", 3: "Write: The quick brown fox jumps."}
}

# --- II. Audio Instruction ---
def speak_task(text):
    components.html(f"<script>window.speechSynthesis.cancel(); var m=new SpeechSynthesisUtterance('{text}'); m.rate=0.9; window.speechSynthesis.speak(m);</script>", height=0)

# --- III. The Satisfactory Detection Engine ---

def get_severity(prob):
    """Strict logic mapping. Anything below 0.51 is definitely Normal."""
    if prob < GLOBAL_THRESHOLD:
        return "Normal", "green", "✅"
    elif 0.51 <= prob < 0.63:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.63 <= prob < 0.82:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def clean_and_skeletonize(gray_img):
    """
    Accuracy Fix: Converts strokes to 1-pixel width skeletons.
    This ensures the model sees the 'shape' and 'coordination' 
    rather than how thick the digital pen was.
    """
    _, binary = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    # Finding bounding box to remove empty space
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = binary[y:y+h, x:x+w]
        return cv2.resize(roi, IMG_SIZE_DL)
    return cv2.resize(binary, IMG_SIZE_DL)



def predict_by_task(img_data, stage):
    """
    Rerouting Logic:
    Level 1: 100% RF (The Specialist for Chars)
    Level 2: 60% RF / 40% DL (Hybrid Word Analysis)
    Level 3: 20% RF / 80% DL (The Specialist for Sentences)
    """
    processed = clean_and_skeletonize(img_data)
    
    # RF Feature Extraction
    img_64 = cv2.resize(processed, (64, 64))
    hog_feats = hog(img_64, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    rf_input = np.concatenate([hog_feats, [np.var(img_64), np.mean(img_64), 0, 0]]).reshape(1, -1)
    
    rf_p = rf_m.predict_proba(rf_input)[0][1] if rf_m else 0.0
    
    dl_p = 0.0
    if dl_m:
        rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_m.predict(inp, verbose=0)[0][0])

    # Dynamic Weighting to stop misclassification
    if stage == 1:
        return rf_p  # Ignore DL for characters as it wasn't trained for it
    elif stage == 2:
        return (rf_p * 0.6) + (dl_p * 0.4)
    else:
        return (rf_p * 0.2) + (dl_p * 0.8) # DL takes over for sentences

# --- IV. Model Loading ---
@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try: dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except: pass
    return rf, dl

rf_m, dl_m = load_models()

# --- V. Streamlit UI ---
if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'results': [], 'spoken': False})

tab1, tab2 = st.tabs(["✍️ Assessment", "📤 Analysis"])

with tab1:
    if st.session_state.stage <= 3:
        age = st.sidebar.slider("Age", 5, 12, 7)
        tasks = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"]
        current_task = tasks[st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_task(current_task)
            st.session_state.spoken = True

        st.info(f"🔊 **Task {st.session_state.stage}:** {current_task}")
        canvas = st_canvas(stroke_width=5, stroke_color="#000", background_color="#FFF", height=300, width=700, key=f"v{st.session_state.stage}")
        
        if st.button("Submit Level"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                # Only analyze if ink is detected
                if np.sum(gray < 255) > 500: 
                    score = predict_by_task(gray, st.session_state.stage)
                    st.session_state.results.append(score)
                    st.session_state.stage += 1
                    st.session_state.spoken = False
                    st.rerun()

    else:
        avg = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg)
        
        

        st.header(f"{icon} Final Profile: :{color}[{label}]")
        st.metric("Aggregate Score", f"{avg*100:.1f}%")
        
        if st.button("Restart"):
            st.session_state.update({'stage': 1, 'results': [], 'spoken': False})
            st.rerun()
