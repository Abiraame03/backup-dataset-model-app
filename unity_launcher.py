import streamlit as st
import urllib.parse

def show_unity_button(final_result: str = "Normal", aggregate_index: float = 0.0):
    st.markdown("---")
    st.subheader("🕹️ Developer Control")

    project_path = "C:/unity_projects/mild_levels"

    # Encode the path safely for a URL
    encoded_path = urllib.parse.quote(project_path, safe="")

    unity_hub_url = f"unityhub://open?path={encoded_path}"

    st.link_button(
        "🚀 OPEN PROJECT IN UNITY HUB",
        url=unity_hub_url,
        use_container_width=True,
        type="primary"
    )

    st.caption("⚠️ Requires Unity Hub to be installed and running on your local machine.")
