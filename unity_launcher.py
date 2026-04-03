import streamlit as st
import subprocess
import os
import glob

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    # --- CONFIGURATION ---
    # Path where Unity Hub usually installs editors
    hub_editors_path = r"C:\Program Files\Unity\Hub\Editor"
    # Path to your project
    project_path = r"C:\unity_projects\mild_levels"

    # --- AUTO-PATH FINDER ---
    # This looks for ANY Unity.exe inside the Hub folder automatically
    search_pattern = os.path.join(hub_editors_path, "*", "Editor", "Unity.exe")
    found_versions = glob.glob(search_pattern)
    
    unity_exe = found_versions[0] if found_versions else None

    # --- UI RENDER ---
    if st.button("🏗️ OPEN PROJECT ASSETS", use_container_width=True, type="primary"):
        if unity_exe and os.path.exists(project_path):
            try:
                # Command: "Run Unity.exe and load this specific project"
                subprocess.Popen([unity_exe, "-projectPath", project_path])
                st.toast("🚀 Launching Unity Editor...")
            except Exception as e:
                st.error(f"Launch failed: {e}")
        else:
            st.error("❌ Setup Required")
            if not unity_exe:
                st.warning("Unity.exe not found in C:\\Program Files\\Unity\\Hub\\Editor")
            if not os.path.exists(project_path):
                st.warning(f"Project not found at: {project_path}")
