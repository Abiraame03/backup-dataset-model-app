import streamlit as st
import os
import subprocess

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🎮 Start Chocolate World")
    
    # This is the EXACT path from your screenshot
    game_path = r"C:\unity_projects\mild_levels\Assets\game levles.unity"
    
    if st.button("🚀 CLICK TO PLAY", use_container_width=True):
        if os.path.exists(game_path):
            try:
                # This is the command that forces Windows to run the EXE
                subprocess.Popen([game_path])
                st.success("✅ Opening Game... Please wait a moment.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            # If you see this error, it means the file name is slightly different
            st.error("❌ File not found at the location!")
            st.info(f"Looking for: {game_path}")
