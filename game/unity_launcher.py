import streamlit as st
import streamlit.components.v1 as components

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("## 🎮 Play Integrated Game")
    
    # Use the new link GitHub gave you + the folder name where your game is
    # If your game is in a folder named 'game', it looks like this:
    base_url = "https://abiraame03.github.io/backup-dataset-model-app/"
    
    game_url = f"{base_url}?result={final_result}&score={aggregate_index}"

    # This embeds the game perfectly!
    import streamlit.components.v1 as components
    components.iframe(game_url, height=650, scrolling=False)
