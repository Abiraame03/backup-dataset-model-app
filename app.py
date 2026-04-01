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
import json

# --- PAGE CONFIG FIRST ---
st.set_page_config(page_title="Precision Dyslexia Analyzer", layout="wide")

# --- SIDEBAR (ONLY ONCE - FIXED) ---
with st.sidebar:
    st.markdown("### 👤 Player Profile")

    name = st.text_input("Name", "Player", key="player_name")
    gender = st.radio("Gender", ["Male", "Female"], key="player_gender")
    age = st.slider("Age", 5, 12, 7, key="player_age")

# --- MAIN TITLE ---
st.title("🧠 Coordination & Dyslexia Severity Analyzer")
st.markdown("---")

# --- CONFIG ---
RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"

CANVAS_THRESHOLD = 0.549
UPLOAD_THRESHOLD = 0.60
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

# --- AUDIO ---
def speak_text(text):
    components.html(f"""
        <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.9;
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- LOGIC ---
def get_severity(prob, threshold):
    if prob < threshold:
        return "Normal", "green", "✅"
    elif threshold <= prob < (threshold + 0.1):
        return "Mild Dyslexia", "blue", "⚠️"
    elif (threshold + 0.1) <= prob < (threshold + 0.2):
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

def preprocess_image(gray_img):
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        roi = gray_img[y:y+h, x:x+w]
        return cv2.resize(roi, IMG_SIZE_DL)
    return cv2.resize(gray_img, IMG_SIZE_DL)

def ensemble_predict(gray_img, stage):
    proc = preprocess_image(gray_img)

    img_64 = cv2.resize(proc, (64, 64))
    feats = hog(img_64, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    rf_inp = np.concatenate([feats, [np.var(img_64), np.mean(img_64), 0, 0]]).reshape(1, -1)
    rf_p = rf_m.predict_proba(rf_inp)[0][1] if rf_m else 0.0

    dl_p = 0.0
    if dl_m:
        rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb / 255.0, axis=0)
        dl_p = float(dl_m.predict(inp, verbose=0)[0][0])

    if stage == 1:
        score = (rf_p * 0.9 + dl_p * 0.1)
    elif stage == 2:
        score = (rf_p * 0.5 + dl_p * 0.5)
    else:
        score = (rf_p * 0.2 + dl_p * 0.8)

    return score, rf_p, dl_p

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = None
    if os.path.exists(DL_MODEL_PATH):
        try:
            dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False)
        except:
            pass
    return rf, dl

rf_m, dl_m = load_models()

# --- SESSION ---
if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1,
        'results': [],
        'rf_raw': [],
        'dl_raw': [],
        'spoken': False,
        'start_time': None
    })

t1, t2 = st.tabs(["✍️ Assessment Canvas", "📤 External File"])

# ================= TAB 1 =================
with t1:
    if st.session_state.stage <= 3:

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        task_list = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"]
        current_task = task_list[st.session_state.stage]

        if not st.session_state.spoken:
            speak_text(current_task)
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        st.info(f"📝 {current_task}")

        canvas = st_canvas(
            stroke_width=5,
            stroke_color="#000",
            background_color="#FFF",
            height=300,
            width=750,
            key=f"canvas_{st.session_state.stage}"
        )

        if st.button(f"Submit Task {st.session_state.stage}"):
            if canvas.image_data is not None:
                gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

                if np.sum(gray < 255) > 400:
                    final_p, r_p, d_p = ensemble_predict(gray, st.session_state.stage)

                    st.session_state.results.append(final_p)
                    st.session_state.rf_raw.append(r_p)
                    st.session_state.dl_raw.append(d_p)

                    st.session_state.stage += 1
                    st.session_state.spoken = False
                    st.rerun()
                else:
                    st.warning("Draw something first!")

    else:
        # ===== FINAL RESULT =====
        avg_score = np.mean(st.session_state.results)
        label, color, icon = get_severity(avg_score, CANVAS_THRESHOLD)

        st.success(f"### Result: {label} {icon}")

        # ===== UNITY BUTTON =====
        st.divider()
        st.subheader("🎮 Play in Unity")

        if st.button("Play in Unity"):

            path = r"C:/temp/unity_data.json"

            level_map = {
                "Normal": 1,
                "Mild Dyslexia": 2,
                "Moderate Dyslexia": 3,
                "Severe Dyslexia": 1
            }

            level = level_map.get(label, 1)

            data = {
                "name": name,
                "age": age,
                "gender": gender,
                "level": level
            }

            os.makedirs("C:/temp", exist_ok=True)

            with open(path, "w") as f:
                json.dump(data, f)

            st.success("✅ Data sent! Now open Unity and press PLAY")

# ================= TAB 2 =================
with t2:
    st.header("Upload Image")
    up = st.file_uploader("Upload image", type=['png','jpg','jpeg'])

    if up:
        img_arr = np.array(Image.open(up).convert('L'))
        st.image(up)

        if st.button("Analyze"):
            final_p, _, _ = ensemble_predict(img_arr, 3)
            label, color, icon = get_severity(final_p, UPLOAD_THRESHOLD)

            st.success(f"{icon} {label}")
