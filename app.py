import streamlit as st
import numpy as np
import cv2
import joblib
from streamlit_drawable_canvas import st_canvas
import time
import os
import streamlit.components.v1 as components
from skimage.feature import hog

# --- App Configuration ---
st.set_page_config(page_title="Dyslexia Detection & Severity Analyzer", layout="wide")

GENERALIZED_GOAL = "Goal: Improving Hand-Eye Coordination through steady strokes and consistent letter sizing."

# Auditory Task Content
PUZZLES = {
    "Beginner (5-7)": {
        1: "Draw the letters b and d slowly and clearly.",
        2: "Write the word CAT using large capital letters.",
        3: "Write the sentence: The sun is hot."
    },
    "Advanced (8-12)": {
        1: "Draw the letters p, q, b, and d in a single row.",
        2: "Write the word MOUNTAIN in your best handwriting.",
        3: "Write: The quick brown fox."
    }
}

TIME_BENCHMARKS = {5:65, 6:60, 7:55, 8:50, 9:45, 10:40, 11:35, 12:30}

# --- Auditory Helper ---
def speak_task(text):
    components.html(f"""
        <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 0.85; 
        window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# --- Feature Extraction (Matches your Training Logic) ---
def enhance_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img

def extract_dyslexia_features(img, img_size=64):
    img = enhance_image(img)
    img = cv2.resize(img, (img_size, img_size))
    hog_feat = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    
    edges = cv2.Canny(img, 80, 160)
    edge_density = np.sum(edges) / (np.sum(img > 0) + 1)
    intensity_var = np.var(img)
    horiz_proj = np.sum(img, axis=1)
    spacing_var = np.var(horiz_proj)
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    stroke_profile = np.sum(binary > 0, axis=1)
    stroke_width_var = np.var(stroke_profile)

    return np.concatenate([hog_feat, [edge_density, intensity_var, spacing_var, stroke_width_var]])

# --- Session State ---
if 'stage' not in st.session_state:
    st.session_state.update({
        'stage': 1, 
        'data': {"imgs": [], "times": [], "level_results": []}, 
        'start_time': None, 
        'spoken': False
    })

# --- Model Loading ---
@st.cache_resource
def load_model():
    path = "dyslexia_RF_model_mixed_chars_sentences_v3.joblib"
    return joblib.load(path) if os.path.exists(path) else None

model = load_model()

st.title("🧠 Coordination & Auditory Dyslexia Analyzer")
st.info(f"**Goal:** {GENERALIZED_GOAL}")

tab1, tab2 = st.tabs(["✍️ Writing Canvas (Main)", "📤 Upload Image"])

with st.sidebar:
    u_age = st.slider("Select Age", 5, 12, 7)
    bracket = "Beginner (5-7)" if u_age <= 7 else "Advanced (8-12)"
    if st.button("Reset Assessment"):
        st.session_state.update({'stage': 1, 'data': {"imgs": [], "times": [], "level_results": []}, 'start_time': None, 'spoken': False})
        st.rerun()

# --- TAB 1: CANVAS MODE ---
with tab1:
    if st.session_state.stage <= 3:
        task_text = PUZZLES[bracket][st.session_state.stage]
        if not st.session_state.spoken:
            speak_task(f"Level {st.session_state.stage}. {task_text}")
            st.session_state.spoken = True

        st.subheader(f"Level {st.session_state.stage}")
        canvas_result = st_canvas(stroke_width=4, stroke_color="#000", background_color="#FFF", height=300, width=800, key=f"c{st.session_state.stage}")

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()

        if st.button(f"Submit Level {st.session_state.stage}"):
            if canvas_result.image_data is not None and st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                gray = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                # --- APPLY 54% THRESHOLD LOGIC ---
                feats = extract_dyslexia_features(gray).reshape(1, -1)
                probs = model.predict_proba(feats)[0]
                dyslexia_confidence = probs[1] * 100
                
                # Custom threshold check
                is_dyslexic = dyslexia_confidence > 54.0
                label = "Dyslexic" if is_dyslexic else "Normal"
                
                st.session_state.data["imgs"].append(gray)
                st.session_state.data["times"].append(elapsed)
                st.session_state.data["level_results"].append({"label": label, "conf": dyslexia_confidence, "time": elapsed})
                
                st.session_state.stage += 1
                st.session_state.start_time = None
                st.session_state.spoken = False
                st.rerun()

    # --- Display Level Results at Bottom ---
    if len(st.session_state.data["level_results"]) > 0:
        st.divider()
        st.subheader("📊 Individual Level Model Outputs")
        cols = st.columns(3)
        for i, res in enumerate(st.session_state.data["level_results"]):
            with cols[i]:
                color = "red" if res["label"] == "Dyslexic" else "green"
                st.markdown(f"**Level {i+1}:**")
                st.markdown(f"<h3 style='color:{color};'>{res['label']}</h3>", unsafe_allow_html=True)
                st.write(f"Confidence: {res['conf']:.1f}%")

    # --- Final Overall Report ---
    if st.session_state.stage > 3:
        st.divider()
        st.header("🏁 Overall Analysis Report")
        
        # Determine overall detection based on majority
        dyslexic_count = sum(1 for r in st.session_state.data["level_results"] if r["label"] == "Dyslexic")
        overall_dyslexic = dyslexic_count >= 2

        if not overall_dyslexic:
            st.balloons()
            st.success("### Overall Detection: Normal / Non-Dyslexic")
            st.write("Handwriting coordination across all tasks is consistent with healthy development. Severity analysis is not required.")
        else:
            # Move to Severity Analysis only for Dyslexic
            avg_time = sum(st.session_state.data["times"]) / 3
            target = TIME_BENCHMARKS.get(u_age, 30)
            diff = avg_time - target
            
            if diff < 5: severity, color = "Mild Risk", "blue"
            elif 5 <= diff < 15: severity, color = "Moderate Risk", "orange"
            else: severity, color = "Severe Risk", "red"

            

            st.error(f"### Overall Detection: Dyslexic Profile Confirmed")
            st.markdown(f"## Final Severity: :{color}[{severity}]")
            st.write(f"**Parameters:** Age: {u_age} | Avg Time: {round(avg_time, 1)}s | Delay: {round(diff, 1)}s vs Target.")

# --- TAB 2: UPLOAD MODE ---
with tab2:
    up = st.file_uploader("Upload Image", type=['png', 'jpg'])
    if up and model:
        img = cv2.imdecode(np.asarray(bytearray(up.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        st.image(img, width=300)
        if st.button("Predict Upload"):
            u_feats = extract_dyslexia_features(img).reshape(1, -1)
            u_probs = model.predict_proba(u_feats)[0]
            u_conf = u_probs[1] * 100
            
            # Apply 54% threshold
            u_res = "Dyslexic" if u_conf > 54.0 else "Normal"
            
            if u_res == "Dyslexic":
                st.error(f"Result: {u_res} ({u_conf:.1f}%)")
                # Severity with Age Alone for uploads
                if u_age <= 7: sev, col = "Mild Risk", "blue"
                elif u_age <= 10: sev, col = "Moderate Risk", "orange"
                else: sev, col = "Severe Risk", "red"
                st.markdown(f"**Severity (Age {u_age}):** :{col}[{sev}]")
            else:
                st.success(f"Result: {u_res} ({ (100-u_conf) if u_conf < 50 else u_conf :.1f}%)")
