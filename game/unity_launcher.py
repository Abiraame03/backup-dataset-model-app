import streamlit as st
import streamlit.components.v1 as components

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("## 🎮 Play Integrated Game")
    
    # This URL points to the game folder in your GitHub deployment
    # Replace 'username' and 'repo-name' with your actual GitHub info
    # Usually: https://<your-username>.github.io/<your-repo-name>/game/index.html
    
    # FOR NOW, let's use a relative path if you're keeping it in the same repo:
    game_url = f"game/index.html?result={final_result}&score={aggregate_index}"

    # Embed the game
    components.iframe(game_url, height=650, scrolling=False)
