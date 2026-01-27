import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas
import streamlit.components.v1 as components

# --- I. Configuration ---
st.set_page_config(page_title="Precision Dyslexia Analyzer", layout="wide")
st.title("🧠 Coordination & Dyslexia Severity Analyzer")
st.markdown("---")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"
GLOBAL_THRESHOLD = 0.5  
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
        3: "Write: The quick brown fox jumps."
    }
}

# --- II. Audio Instruction Component ---
def speak_text(text):
    """Triggers Browser Text-to-Speech"""
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
    """Strict mapping: Below 51% is strictly Normal."""
    if prob < GLOBAL_THRESHOLD:
        return "Normal", "green", "✅"
    elif GLOBAL_THRESHOLD <= prob < 0.65:
        return "Mild Dyslexia", "blue", "⚠️"
    elif 0.65 <= prob < 0.85:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def preprocess_image(gray_img):
    """Crops and centers writing to remove background noise."""
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = gray_img[y:y+h, x:x+w]
        return cv2.resize(roi, IMG_SIZE_DL)
    return cv2.resize(gray_img, IMG_SIZE_DL)



def ensemble_predict(gray_img, stage):
    """
    Accuracy Routing:
    - Stage 1 (Chars): 90% RF / 10% DL (DL is not trained for chars)
    - Stage 3 (Sentences): 20% RF / 80% DL (DL is the specialist)
    """
    proc = preprocess_image(gray_img)
    
    # RF Logic
    img_64 = cv2.resize(proc, (64, 64))
    feats = hog(img_64, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    rf_inp = np.concatenate([feats, [np.var(img_64), np.mean(img_64), 0, 0]]).reshape(1, -1)
    rf_p = rf_m.predict_proba(rf_inp)[0][1] if rf_m else 0.0

    # DL Logic
    dl_p = 0.0
    if dl_m:
        rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_m.predict(inp, verbose=0)[0][0])

    # Dynamic Weighting
    if stage == 1:
        score = (rf_p * 0.9 + dl_p * 0.1)
    elif stage == 2:
        score = (rf_p * 0.5 + dl_p * 0.5)
    else:
        score = (rf_p * 0.2 + dl_p * 0.8)
        
    return score, rf_p, dl_p

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

# --- V. UI Workflow ---

if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 
        'results': [], 
        'rf_raw': [], 
        'dl_raw': [], 
        'spoken': False
    })

t1, t2 = st.tabs(["✍️ Assessment Canvas", "📤 External File"])

with t1:
    if st.session_state.stage <= 3:
        age = st.sidebar.slider("Age", 5, 12, 7)
        task_list = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"]
        current_task = task_list[st.session_state.stage]
        
        if not st.session_state.spoken:
            speak_text(f"Task {st.session_state.stage}. {current_task}")
            st.session_state.spoken = True

        col_text, col_audio = st.columns([4, 1])
        with col_text:
            st.subheader(f"Level {st.session_state.stage}")
            st.info(f"📝 **Task:** {current_task}")
        with col_audio:
            st.write("") # Spacer
            if st.button("🔊 Replay"):
                speak_text(current_task)

        canvas = st_canvas(stroke_width=5, stroke_color="#000", background_color="#FFF", height=300, width=750, key=f"c{st.session_state.stage}")
        
        if st.button(f"Submit Task {st.session_state.stage}", use_container_width=True):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                # Ensure user actually wrote something
                if np.sum(gray < 255) > 400:
                    final_p, r_p, d_p = ensemble_predict(gray, st.session_state.stage)
                    st.session_state.results.append(final_p)
                    st.session_state.rf_raw.append(r_p)
                    st.session_state.dl_raw.append(d_p)
                    st.session_state.stage += 1
                    st.session_state.spoken = False
                    st.rerun()
                else:
                    st.warning("Canvas is empty. Please draw the task.")

    else:
        # --- OVERALL SUMMARY SECTION ---
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score)
        test_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if label == "Normal":
            st.balloons()
            st.success(f"### Final Result: {label} {icon}")
        else:
            st.error(f"### Final Result: {label} {icon}")

        st.write(f"🕒 **Completion Time:** {test_time}")
        
        

        # Model Prediction Breakdown
        with st.expander("🔍 Detailed Model Performance Breakdown"):
            st.write("This table shows how each model performed at different complexity levels.")
            summary_data = []
            for i in range(3):
                summary_data.append({
                    "Level": i+1,
                    "RF Prediction (General)": f"{st.session_state.rf_raw[i]*100:.1f}%",
                    "DL Prediction (Sentence)": f"{st.session_state.dl_raw[i]*100:.1f}%",
                    "Weighted Ensemble": f"{st.session_state.results[i]*100:.1f}%"
                })
            st.table(summary_data)

        st.divider()
        st.metric("Aggregate Prediction Index", f"{avg_score*100:.1f}%", 
                  delta=f"{avg_score - GLOBAL_THRESHOLD:.2f}", delta_color="inverse")
        
        st.write(f"**Final Summary:** Based on the weighted analysis of both the Random Forest (Character-trained) and Deep Learning (Sentence-trained) models, the user's handwriting profile indicates a **{label}** pattern.")

        if st.button("Start New Assessment"):
            st.session_state.update({'stage': 1, 'results': [], 'rf_raw': [], 'dl_raw': [], 'spoken': False})
            st.rerun()

with t2:
    st.header("Upload Image Analysis")
    up = st.file_uploader("Upload a photo of written text", type=['png', 'jpg', 'jpeg'])
    if up:
        img_arr = np.array(Image.open(up).convert('L'))
        st.image(up, width=400)
        if st.button("Run Sentence Analysis"):
            # Assume Stage 3 logic for general sentence uploads
            final_p, r_p, d_p = ensemble_predict(img_arr, stage=3)
            label, color, icon = get_severity(final_p)
            st.markdown(f"## {icon} Detection: :{color}[{label}]")
            st.progress(final_p)
            st.write(f"Combined Certainty: **{final_p*100:.1f}%**")
