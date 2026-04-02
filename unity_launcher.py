"""
unity_launcher.py
-----------------
FILE LOCATION: Save this in your GitHub repo, same folder as app.py

HOW TO USE IN app.py — add these lines at the BOTTOM of app.py:
─────────────────────────────────────────────────────────────────
    from unity_launcher import show_unity_button
    show_unity_button(final_result, aggregate_index)
─────────────────────────────────────────────────────────────────
Replace `final_result` with your result variable (e.g. "Mild Dyslexia")
Replace `aggregate_index` with your score variable (e.g. 56.3)
"""

import streamlit as st
import json


def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):

    result_data = {
        "dyslexia_result": str(final_result),
        "aggregate_index": round(float(aggregate_index), 1)
    }
    result_json = json.dumps(result_data, indent=2)

    st.markdown("---")
    st.markdown("## 🎮 Play in Unity")
    st.write("Launch the Chocolate World dyslexia game with your result loaded.")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Step 1: Download Result File",
            data=result_json,
            file_name="dyslexia_result.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        import webbrowser
        if st.button("🎮 Step 2: Play Unity Game", use_container_width=True):
            webbrowser.open("file:///C:/unity_projects/index.html")
            st.success("✅ Opening Unity Game in Browser...")
    st.info(
        "**First time?** Run the `ChocolateWorld_Launcher.bat` file on your PC once "
        "to register the game launcher. After that, Step 2 will open Unity automatically."
    )
