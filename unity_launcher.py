import streamlit as st
import subprocess
import os
import glob

def show_unity_button(final_result: str = "Normal", aggregate_index: float = 0.0):
    # 1. The folder where your Unity versions are installed
    hub_path = r"C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Unity.exe"
    
    # 2. Your specific project folder
    project_path = r"C:\unity_projects\mild_levels"

    # AUTO-FINDER: This looks for any Unity.exe on your computer
    search = os.path.join(hub_path, "*", "Editor", "Unity.exe")
    found = glob.glob(search)
    unity_exe = found[0] if found else None

    if st.button("🚀 OPEN UNITY ASSETS NOW", type="primary", use_container_width=True):
        if unity_exe and os.path.exists(project_path):
            try:
                # The '-projectPath' command tells Unity to skip the Hub and open the project
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.toast("Opening Unity Editor... check your taskbar!")
            except Exception as e:
                st.error(f"Launch Error: {e}")
        else:
            st.error("❌ Path Error: Unity or Project folder not found.")
            st.info(f"Looking for Project at: {project_path}")
