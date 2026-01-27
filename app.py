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
GLOBAL_THRESHOLD = 0.51  # Base detection threshold
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
        3: "Write: The quick brown fox."
    }
}

# --- II. Audio Task Component ---
def speak_task(text):
    """Triggers Browser Text-to-Speech for instructions."""
    components.html(f"""
        <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.85;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- III. Logic & Accuracy Engine ---

def get_severity_label(prob):
    """
    Classifies based on the 51% threshold. 
    If below 0.51, it is strictly 'Normal'.
    """
    if prob < GLOBAL_THRESHOLD:
        return "Normal", "green", "✅"
    elif GLOBAL_THRESHOLD <= prob < 0.65:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.65 <= prob < 0.85:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def preprocess_image(gray_img):
    """Removes noise and crops to the ink for better model accuracy."""
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add small padding
        pad = 10
        cropped = gray_img[max(0, y-pad):min(gray_img.shape[0], y+h+pad), 
                          max(0, x-pad):min(gray_img.shape[1], x+w+pad)]
        return cv2.resize(cropped, IMG_SIZE_DL)
    return cv2.resize(gray_img, IMG_SIZE_DL)

def extract_rf_features(img):
    """Extracts HOG features specifically for the RF model."""
    img_64 = cv2.resize(img, (64, 64))
    features = hog(img_64, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    placeholders = [np.var(img_64), np.mean(img_64), 0, 0] 
    return np.concatenate([features, placeholders]).reshape(1, -1)



def ensemble_prediction(gray_img, stage):
    """
    DYNAMIC ROUTING:
    - Stage 1: RF (85%) | DL (15%) -> RF knows characters better.
    - Stage 2: RF (50%) | DL (50%) -> Mixed word complexity.
    - Stage 3: RF (20%) | DL (80%) -> DL is the sentence specialist.
    """
    proc_img = preprocess_image(gray_img)
    
    # Get RF Probability
    rf_p = rf_m.predict_proba(extract_rf_features(proc_img))[0][1] if rf_m else 0.0
    
    # Get DL Probability
    dl_p = 0.0
    if dl_m:
        rgb = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_m.predict(inp, verbose=0)[0][0])
    
    # Weighted calculation based on Stage
    if stage == 1:
        score = (rf_p * 0.85) + (dl_p * 0.15)
    elif stage == 2:
        score = (rf_p * 0.50) + (dl_p * 0.50)
    else:
        score = (rf_p * 0.20) + (dl_p * 0.80)
        
    return score

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

# --- V. UI Implementation ---

if 'stage' not in st.session_state:
    st.session_state.update({'stage': 1, 'scores': [], 'spoken': False})

tab1, tab2 = st.tabs(["✍️ Handwriting Assessment", "📤 Image Upload"])

with tab1:
    if st.session_state.stage <= 3:
        age = st.sidebar.select_slider("Select Age", options=list(range(5, 13)), value=7)
        age_group = "Beginner (5-7)" if age <= 7 else "Advanced (8-12)"
        task_text = PUZZLES[age_group][st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_task(task_text)
            st.session_state.spoken = True

        st.subheader(f"Step {st.session_state.stage}: {age_group}")
        st.info(f"🔊 **Instruction:** {task_text}")
        
        canvas = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", 
                          height=300, width=700, key=f"canv_{st.session_state.stage}")
        
        if st.button(f"Next Step", use_container_width=True):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                # Check if user actually wrote something
                if np.mean(gray) < 250: 
                    final_p = ensemble_prediction(gray, st.session_state.stage)
                    st.session_state.scores.append(final_p)
                    st.session_state.stage += 1
                    st.session_state.spoken = False
                    st.rerun()
                else:
                    st.warning("Please write something on the canvas first!")

    else:
        # Final Severity Assessment
        avg_score = np.mean(st.session_state.scores)
        label, color, icon = get_severity_label(avg_score)
        
        st.header("📊 Final Handwriting Profile")
        
        if label == "Normal":
            st.balloons()
            st.success(f"### Result: {label} {icon}")
            st.write("Your handwriting patterns are within the typical range for your age group.")
        else:
            st.error(f"### Result: {label} {icon}")
            st.write(f"The system has identified markers consistent with **{label}** based on spatial consistency and stroke patterns.")

        

        st.divider()
        st.metric("Risk Index", f"{avg_score*100:.1f}%", delta=f"{avg_score - GLOBAL_THRESHOLD:.2f}", delta_color="inverse")
        
        if st.button("Restart Test"):
            st.session_state.update({'stage': 1, 'scores': [], 'spoken': False})
            st.rerun()

with tab2:
    st.header("Upload Written Sample")
    up = st.file_uploader("Upload a clear photo of a sentence", type=['png', 'jpg', 'jpeg'])
    if up:
        img_file = Image.open(up).convert('L')
        st.image(up, width=400)
        if st.button("Analyze Uploaded Sentence"):
            # Since this is a sentence, we route using Stage 3 logic (DL high weight)
            score = ensemble_prediction(np.array(img_file), stage=3)
            label, color, icon = get_severity_label(score)
            st.markdown(f"## {icon} Detection: :{color}[{label}]")
            st.progress(score)
            st.write(f"System Certainty: {score*100:.1f}%")
