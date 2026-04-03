import streamlit as st
import subprocess
import os

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🏗️ Open Project in Unity Editor")

    # 1. LIST OF COMMON UNITY PATHS (It will try each one until it finds yours)
    possible_unity_paths = [
        r"C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Unity.exe",
        r"C:\Program Files\Unity\Editor\Unity.exe",
        # Add any other paths if you installed Unity somewhere else
    ]
    
    # 2. YOUR PROJECT PATH
    project_path = r"C:\unity_projects\mild_levels"

    # Find which Unity path actually exists on your PC
    unity_exe = None
    for path in possible_unity_paths:
        if os.path.exists(path):
            unity_exe = path
            break

    if st.button("🏗️ Open Assets in Unity", use_container_width=True):
        if unity_exe and os.path.exists(project_path):
            try:
                # Opens Unity directly to your 'mild_levels' project
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.success("✅ Unity is launching your project Assets...")
            except Exception as e:
                st.error(f"Launch failed: {e}")
        else:
            st.error("❌ PATH ERROR")
            if not unity_exe:
                st.warning("I cannot find Unity.exe. Please check where Unity is installed on your C: drive.")
            if not os.path.exists(project_path):
                st.warning(f"I cannot find your project at: {project_path}")
