import streamlit as st
import subprocess
import os

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🛠️ Open Project in Unity Editor")

    # 1. PATH TO YOUR UNITY EDITOR (Check your version number!)
    # Common path: C:\Program Files\Unity\Hub\Editor\2022.3.x\Editor\Unity.exe
    unity_exe = r"C:\Program Files\Unity\Hub\Editor\6000.3.2f1\Editor" 
    
    # 2. PATH TO YOUR PROJECT FOLDER
    project_path = r"C:\unity_projects\mild_levels"

    if st.button("🏗️ Open Project in Editor", use_container_width=True):
        if os.path.exists(unity_exe) and os.path.exists(project_path):
            try:
                # This command tells Unity to open the specific project folder
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.success("✅ Unity is launching your project...")
            except Exception as e:
                st.error(f"Failed to launch Unity: {e}")
        else:
            st.error("❌ Path Error!")
            st.info(f"Checking Unity: {os.path.exists(unity_exe)}")
            st.info(f"Checking Project: {os.path.exists(project_path)}")
            st.warning("Please verify the 'unity_exe' path matches your version of Unity.")
