import streamlit as st
import streamlit.components.v1 as components
import subprocess
import time
import requests
import os

def start_game_server():
    """Starts a local server in the background if it's not already running."""
    try:
        # Check if server is already running on port 8000
        requests.get("http://localhost:8000", timeout=0.1)
    except:
        # Path to your Unity build folder
        game_path = "C:/unity_projects"
        
        if os.path.exists(game_path):
            # Start the Python server silently in the background
            subprocess.Popen(
                ["python", "-m", "http.server", "8000"], 
                cwd=game_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1) # Wait for server to boot

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    # 1. Start the server automatically
    start_game_server()

    st.markdown("---")
    st.markdown("## 🎮 Play Integrated Game")
    st.write("The game below is now connected to your AI analysis.")

    # 2. Create the URL with the AI data attached
    # This sends your results directly TO the Unity game
    game_url = f"http://localhost:8000/index.html?result={final_result}&score={aggregate_index}"

    # 3. Embed the game directly in the page (No new tabs needed!)
    components.iframe(game_url, height=650, scrolling=False)

    st.success(f"✅ Game loaded with Result: {final_result} ({aggregate_index}%)")
