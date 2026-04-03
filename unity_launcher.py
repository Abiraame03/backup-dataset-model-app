import streamlit as st
import subprocess
import os

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🏗️ Open Project in Unity Editor")

    # PASTE YOUR PATH FROM STEP 1 HERE:
    unity_exe = r"C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor\Unity.exe"
    
    # YOUR PROJECT FOLDER
    project_path = r"C:\unity_projects\mild_levels"

    if st.button("🏗️ Open Assets in Unity", use_container_width=True):
        # Validation Check
        unity_exists = os.path.exists(unity_exe)
        project_exists = os.path.exists(project_path)
        
        if unity_exists and project_exists:
            try:
                # This opens the Unity Editor and tells it which project to load
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.success("✅ Unity is opening your project Assets!")
            except Exception as e:
                st.error(f"Launch failed: {e}")
        else:
            st.error("❌ PATH ERROR")
            if not unity_exists:
                st.warning(f"Unity.exe not found at: {unity_exe}")
            if not project_exists:
                st.warning(f"Project folder not found at: {project_path}")
