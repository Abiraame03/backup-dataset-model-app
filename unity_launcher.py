import streamlit as st
import subprocess
import os

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🏗️ Unity Editor Control")

    # 1. AUTO-DETECT UNITY (Checks the most likely locations)
    possible_paths = [
        r"C:\Program Files\Unity Hub\Unity Hub.exe",
        r"C:\Program Files\Unity\Hub\Editor\2022.3.10f1\Editor\Unity.exe",
        r"C:\Program Files\Unity\Editor\Unity.exe"
        r"C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Unity.exe"
    ]
    
    unity_exe = next((p for p in possible_paths if os.path.exists(p)), None)
    project_path = r"C:\unity_projects\mild_levels"

    # 2. THE BUTTON (We use a bright color to make sure you see it)
    if st.button("🚀 OPEN UNITY ASSETS NOW", use_container_width=True):
        if unity_exe and os.path.exists(project_path):
            try:
                # Forces Unity to open the project folder directly
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.success("✅ Unity is launching... check your taskbar!")
            except Exception as e:
                st.error(f"System blocked launch: {e}")
        else:
            st.error("❌ PATH ERROR: Unity or Project not found.")
            if not unity_exe:
                st.warning("Could not find Unity.exe. Please verify your Unity version folder.")
            if not os.path.exists(project_path):
                st.warning(f"Project folder missing at: {project_path}")

    st.caption("Note: This button only works when running Streamlit on your local computer.")
