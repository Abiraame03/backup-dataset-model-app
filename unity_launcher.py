import streamlit as st
import os
import subprocess

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("### 🎮 Launch Chocolate World")
    
    # This matches your folder name and exe name exactly
    game_path = r"C:\unity_projects\mild_levels\Build\mild_levels.exe"
    
    if st.button("🚀 Launch Local Game", use_container_width=True):
        if os.path.exists(game_path):
            try:
                # This opens the EXE and passes the AI results as arguments
                subprocess.Popen([
                    game_path, 
                    f"--result={final_result}", 
                    f"--score={aggregate_index}"
                ])
                st.success("🚀 Game is opening! Check your taskbar.")
            except Exception as e:
                st.error(f"Error launching: {e}")
        else:
            st.error(f"❌ Could not find game at: {game_path}")
            st.info("Ensure your Unity build is in the folder: Build and named: mild_levels.exe")
