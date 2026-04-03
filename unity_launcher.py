import streamlit as st
import os
import subprocess

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("### 🎮 Launch Chocolate World")
    
    # Path to your local build
    game_path = r"C:\unity_projects\mild_levels\Build\mild_levels.exe"
    
    if st.button("🚀 Launch Local Game", use_container_width=True):
        if os.path.exists(game_path):
            try:
                # This opens the EXE directly and passes the AI score
                subprocess.Popen([game_path, f"--result={final_result}", f"--score={aggregate_index}"])
                st.success("🚀 Game is opening! Check your taskbar.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error(f"❌ File not found at: {game_path}")
            st.info("Make sure your Unity build is in C:\\unity_projects\\mild_levels\\Build\\ and named mild_levels.exe")
