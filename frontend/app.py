"""Main Streamlit frontend application for Document QA RAG system.

Provides web UI for:
- Chat interface for querying documents
- Session management (new, load, delete)
- Configuration selection for retrieval settings
- Display of retrieved figures from papers
"""

import streamlit as st
from components.sidebar import render_sidebar
from components.chat import render_chat

st.set_page_config(
    page_title="Document QA",
    page_icon="🔮",
    layout="wide",
)

from pathlib import Path

css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
else:
    # fallback: no-op if file missing
    pass

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

thread_id, config_id = render_sidebar()
render_chat(thread_id, config_id)