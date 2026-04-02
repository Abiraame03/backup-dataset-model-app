import streamlit as st
import streamlit.components.v1 as components

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("## 🎮 Play Integrated Game")
    
    # 1. This is the exact URL to your folder on GitHub Pages
    base_url = "https://abiraame03.github.io/backup-dataset-model-app/game/index.html"
    
    # 2. This attaches your results so Unity can read them
    game_url = f"{base_url}?result={final_result}&score={aggregate_index}"

    # 3. This displays the game window
    components.iframe(game_url, height=700, scrolling=False)
    
    st.info(f"Analysis sent to game: {final_result} ({aggregate_index}%)")
