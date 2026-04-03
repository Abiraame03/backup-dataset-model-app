"""
unity_launcher.py
─────────────────────────────────────────────────────────────────
Add this file to your GitHub repo (same folder as app.py).

Then add TWO lines to the BOTTOM of your app.py:

    from unity_launcher import show_unity_section
    show_unity_section(final_result, aggregate_index)

Replace final_result and aggregate_index with YOUR variable names.
─────────────────────────────────────────────────────────────────
"""

import streamlit as st


def show_unity_section(final_result: str = "Unknown", aggregate_index: float = 0.0):

    bat = (
        "@echo off\n"
        f"echo Opening Chocolate World - {final_result}\n"
        'start "" "C:\\unity_projects\\mild_levels\\Build\\ChocolateWorld.exe"\n'
        "exit\n"
    )

    st.markdown("---")
    st.markdown("### 🎮 Play in Unity")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.download_button(
            label="🎮 Download & Open Unity Game",
            data=bat,
            file_name="OpenChocolateWorld.bat",
            mime="text/plain",
            use_container_width=True,
            type="primary",
        )

    with col2:
        st.info(
            f"**Result:** {final_result}  \n"
            f"**Score:** {round(float(aggregate_index), 1)}%"
        )

    st.caption(
        "Click the button → file downloads → double-click OpenChocolateWorld.bat → game opens."
    )
