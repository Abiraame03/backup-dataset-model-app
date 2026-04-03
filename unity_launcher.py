import streamlit as st
import os

def show_unity_button(final_result: str = "Unknown", aggregate_index: float = 0.0):
    st.markdown("---")
    st.markdown("### 🎮 Launch Chocolate World")
    st.info(f"Analysis Profile: **{final_result}** ({aggregate_index}%)")

    # Since the app is in the cloud, it can't "click" a file on your C: drive directly.
    # We will provide a clear instruction for the user to launch their local build.
    
    if st.button("🚀 Open Local Game Instance", use_container_width=True):
        # This tries to trigger the local custom protocol we set up earlier
        launch_url = f"unitylauncher://?result={final_result}&index={aggregate_index}"
        st.markdown(f'<meta http-equiv="refresh" content="0; url={launch_url}">', unsafe_allow_html=True)
        st.success("Check your taskbar! The game is launching...")
