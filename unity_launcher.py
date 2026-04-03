import streamlit as st
import subprocess
import os

def show_unity_button(final_result: str = "Normal", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🕹️ Developer Control")

    # Use forward slashes / to avoid Windows path errors in Python
    unity_exe = "C:/Program Files/Unity/Hub/Editor/6000.3.2f1/Editor/Unity.exe"
    project_path = "C:/unity_projects/mild_levels"

    if st.button("🚀 OPEN PROJECT IN UNITY", type="primary", use_container_width=True):
        u_exists = os.path.exists(unity_exe)
        p_exists = os.path.exists(project_path)
        
        if u_exists and p_exists:
            try:
                # This opens the editor and loads your project directly
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.toast("Launching Unity... please wait.")
            except Exception as e:
                st.error(f"Launch Error: {e}")
        else:
            st.error("❌ PATH ERROR")
            if not u_exists:
                st.warning(f"Unity.exe NOT found at: {unity_exe}")
            if not p_exists:
                st.warning(f"Project folder NOT found at: {project_path}")
