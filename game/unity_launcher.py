import streamlit as st
import streamlit.components.v1 as components

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("## 🎮 Play Integrated Game")
    st.write("The game below is now connected to your AI analysis.")

    # 1. Your Live GitHub Pages Link
    base_url = "https://abiraame03.github.io/backup-dataset-model-app/game/index.html"

    # 2. Attach the results to the URL so Unity can read them
    # We use 'result' and 'score' as the keys
    game_url = f"{base_url}?result={final_result}&score={aggregate_index}"

    # 3. Embed the game directly into the Streamlit page
    # Height 650 is usually perfect for Unity WebGL
    components.iframe(game_url, height=650, scrolling=False)

    st.info(f"🕹️ Current Profile: {final_result} | Intensity: {aggregate_index}%")
