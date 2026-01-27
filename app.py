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

# --- II. Robust Audio Instruction Component ---
def speak_text(text):
    """Triggers the browser's native Text-to-Speech engine."""
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.9; 
        msg.pitch = 1;
        window.speechSynthesis.cancel(); 
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- III. Logic & Accuracy Engine ---

def preprocess_handwriting(gray_img):
    """Crops image to the ink to improve prediction accuracy."""
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = gray_img[y:y+h, x:x+w]
        return cv2.resize(cropped, (IMG_SIZE_DL[0], IMG_SIZE_DL[1]))
    return cv2.resize(gray_img, (IMG_SIZE_DL[0], IMG_SIZE_DL[1]))

def get_severity(prob):
    """Refined severity bands based on 51% threshold."""
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
    features = hog(img_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    placeholders = [np.var(img_res), np.mean(img_res), 0, 0] 
    return np.concatenate([features, placeholders]).reshape(1, -1)



def predict_all(gray_img, rf_model, dl_model, use_dl=True):
    processed_img = preprocess_handwriting(gray_img)
    rf_p, dl_p = 0.0, 0.0
    if rf_model:
        rf_p = rf_model.predict_proba(extract_features(processed_img))[0][1]
    if dl_model and use_dl:
        rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_model.predict(inp, verbose=0)[0][0])
    
    if use_dl:
        combined = (rf_p * 0.4 + dl_p * 0.6)
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
        age = st.sidebar.slider("Age of Participant", 5, 12, 7)
        task_list = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"]
        current_task = task_list[st.session_state.stage]
        
        # Automatic Audio Trigger
        if not st.session_state.spoken:
            speak_text(f"Level {st.session_state.stage}. {current_task}")
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(f"📋 **Task:** {current_task}")
        with col2:
            if st.button("🔊 Replay"):
                speak_text(current_task)

        canvas = st_canvas(
            stroke_width=4, 
            stroke_color="#000", 
            background_color="#FFF", 
            height=350, 
            width=750, 
            key=f"canvas_stage_{st.session_state.stage}"
        )
        
        if st.button(f"Submit Level {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                score = predict_all(gray, rf_m, dl_m, use_dl=(st.session_state.stage >= 2))
                st.session_state.results.append(score)
                st.session_state.stage += 1
                st.session_state.spoken = False 
                st.rerun()
    else:
        # Final Report
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score)
        
        

        if label == "Normal":
            st.balloons()
            st.success(f"### Assessment Result: {label} {icon}")
        else:
            st.error(f"### Assessment Result: {label} {icon}")
        
        st.metric("Final Risk Probability", f"{avg_score*100:.1f}%")
        if st.button("Restart Assessment"):
            st.session_state.update({'stage': 1, 'results': [], 'spoken': False})
            st.rerun()

with t2:
    st.header("Single Image Analysis")
    up = st.file_uploader("Upload handwriting photo", type=['png', 'jpg', 'jpeg'])
    if up:
        img = np.array(Image.open(up).convert('L'))
        st.image(up, width=300, caption="Uploaded Sample")
        if st.button("Analyze Upload"):
            score = predict_all(img, rf_m, dl_m, use_dl=True)
            label, color, icon = get_severity(score)
            st.markdown(f"### {icon} Result: :{color}[{label}]")
            st.progress(score)
            st.write(f"Confidence: {score*100:.1f}%")
