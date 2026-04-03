import streamlit as st
import subprocess
import os

def show_unity_button(final_result: str = "Normal", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🕹️ Developer Control")

    # 1. The EXACT path to your Unity Editor (From your screenshots)
    unity_exe = r"C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Unity.exe"
    
    # 2. Your specific project folder
    project_path = r"C:\unity_projects\mild_levels"

    if st.button("🚀 OPEN PROJECT IN UNITY", type="primary", use_container_width=True):
        if os.path.exists(unity_exe) and os.path.exists(project_path):
            try:
                # This opens the editor and loads your project directly
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.toast("Launching Unity... please wait.")
            except Exception as e:
                st.error(f"Launch Error: {e}")
        else:
            st.error("❌ Path Error!")
            if not os.path.exists(unity_exe):
                st.warning(f"Unity not found at: {unity_exe}")
            if not os.path.exists(project_path):
                st.warning(f"Project not found at: {project_path}")
