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

# --- CONFIG ---
st.set_page_config(page_title="Dyslexia Analyzer", layout="wide")

# --- SIDEBAR PROFILE ---
with st.sidebar:
    st.markdown("### 👤 Player Profile")
    st.session_state["player_name"] = st.text_input("Name", "Player")
    st.session_state["player_gender"] = st.radio("Gender", ["Male", "Female"])
    st.session_state["player_age"] = st.slider("Age", 5, 12, 7)

st.title("🧠 Dyslexia Severity Analyzer")
st.markdown("---")

RF_MODEL_PATH = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
DL_MODEL_PATH = "mobilenetv2_bilstm_final.h5"

CANVAS_THRESHOLD = 0.549
UPLOAD_THRESHOLD = 0.60
IMG_SIZE_DL = (160, 160)

PUZZLES = {
    "Beginner (5-7)": {
        1: "Draw the letters b and d slowly.",
        2: "Write the word CAT.",
        3: "Write: The sun is hot."
    },
    "Advanced (8-12)": {
        1: "Draw p, q, b, d.",
        2: "Write MOUNTAIN.",
        3: "Write: The quick brown fox."
    }
}

# --- SPEECH ---
def speak_text(text):
    components.html(f"""
    <script>
    var msg = new SpeechSynthesisUtterance('{text}');
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# --- SEVERITY ---
def get_severity(prob, threshold):
    if prob < threshold:
        return "Normal", "green", "✅"
    elif prob < threshold + 0.1:
        return "Mild Dyslexia", "blue", "⚠️"
    elif prob < threshold + 0.2:
        return "Moderate Dyslexia", "orange", "🟠"
    else:
        return "Severe Dyslexia", "red", "🔴"

# --- IMAGE PROCESS ---
def preprocess(gray):
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(th)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE_DL)
    return cv2.resize(gray, IMG_SIZE_DL)

# --- PREDICT ---
def ensemble(gray, stage):
    proc = preprocess(gray)

    img64 = cv2.resize(proc, (64, 64))
    feats = hog(img64, pixels_per_cell=(8,8), cells_per_block=(2,2))
    rf_in = np.concatenate([feats, [np.var(img64), np.mean(img64), 0, 0]]).reshape(1,-1)
    rf_p = rf_m.predict_proba(rf_in)[0][1] if rf_m else 0

    dl_p = 0
    if dl_m:
        rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        inp = np.expand_dims(rgb/255.0, axis=0)
        dl_p = float(dl_m.predict(inp)[0][0])

    if stage == 1: return rf_p*0.9 + dl_p*0.1, rf_p, dl_p
    if stage == 2: return rf_p*0.5 + dl_p*0.5, rf_p, dl_p
    return rf_p*0.2 + dl_p*0.8, rf_p, dl_p

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    rf = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
    dl = tf.keras.models.load_model(DL_MODEL_PATH, compile=False) if os.path.exists(DL_MODEL_PATH) else None
    return rf, dl

rf_m, dl_m = load_models()

# --- SESSION ---
if 'stage' not in st.session_state:
    st.session_state.update({
        'stage':1,'results':[],'rf':[],'dl':[],'spoken':False,'start':None
    })

# ============================
# MAIN FLOW
# ============================

if st.session_state.stage <= 3:

    if st.session_state.start is None:
        st.session_state.start = time.time()

    age = st.session_state["player_age"]
    tasks = PUZZLES["Beginner (5-7)" if age <= 7 else "Advanced (8-12)"]
    task = tasks[st.session_state.stage]

    if not st.session_state.spoken:
        speak_text(task)
        st.session_state.spoken = True

    st.subheader(f"Level {st.session_state.stage}")
    st.info(task)

    canvas = st_canvas(height=300, width=700, key=f"c{st.session_state.stage}")

    if st.button("Submit"):
        if canvas.image_data is not None:
            gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

            if np.sum(gray < 255) > 400:
                p, r, d = ensemble(gray, st.session_state.stage)
                st.session_state.results.append(p)
                st.session_state.rf.append(r)
                st.session_state.dl.append(d)
                st.session_state.stage += 1
                st.session_state.spoken = False
                st.rerun()
            else:
                st.warning("Draw something!")

# ============================
# FINAL RESULT
# ============================

else:
    avg = np.mean(st.session_state.results)
    label, color, icon = get_severity(avg, CANVAS_THRESHOLD)

    st.success(f"### Result: {label} {icon}")

    # ================= UNITY BUTTON =================
    st.subheader("🎮 Play in Unity")

    if st.button("🎮 Open & Play in Unity"):

        path = r"C:/temp/unity_data.json"

        level_map = {
            "Normal": 1,
            "Mild Dyslexia": 2,
            "Moderate Dyslexia": 3,
            "Severe Dyslexia": 1
        }

        data = {
            "name": st.session_state["player_name"],
            "age": st.session_state["player_age"],
            "gender": st.session_state["player_gender"],
            "level": level_map.get(label, 1)
        }

        os.makedirs("C:/temp", exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f)

        st.success("✅ Data sent to Unity!")
        st.info("👉 Open Unity and press ▶ Play")

    # RESET
    if st.button("Start New Assessment"):
        st.session_state.update({
            'stage':1,'results':[],'rf':[],'dl':[],'spoken':False,'start':None
        })
        st.rerun()
        
