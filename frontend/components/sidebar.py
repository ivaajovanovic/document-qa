"""Sidebar UI for session management and configuration selection.

Provides:
- Button to create new chat sessions
- List of previous sessions with quick-access buttons
- Delete session functionality
- Default retrieval configuration
"""

import streamlit as st
from ..api_client import get_sessions, new_session, delete_session, get_history

DEFAULT_CONFIG_ID = "config_multimodal_k10_rrf60"


def render_sidebar() -> tuple[str, str]:
    """Render sidebar with session management and return active session info.
    
    Returns:
        Tuple of (thread_id, config_id) for the active session.
        thread_id is None if no session is selected.
    """
    with st.sidebar:
        st.title("🔮 Document QA")
        st.caption("Academic Research Assistant")

        st.divider()

        if st.button("＋ New Chat", use_container_width=True):
            thread_id = new_session()
            st.session_state.thread_id = thread_id
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.subheader("Chat History")
        try:
            sessions = get_sessions()

            if not sessions:
                st.caption("No previous chats.")

            for session in sessions:
                col1, col2 = st.columns([5, 1])

                with col1:
                    is_active = session["thread_id"] == st.session_state.get("thread_id")
                    label = f"{'▶ ' if is_active else ''}{session['name']}"
                    if st.button(label, key=session["thread_id"], use_container_width=True):
                        st.session_state.thread_id = session["thread_id"]
                        st.session_state.messages = get_history(session["thread_id"])
                        st.rerun()

                with col2:
                    if st.button("🗑", key=f"del_{session['thread_id']}"):
                        delete_session(session["thread_id"])
                        if st.session_state.get("thread_id") == session["thread_id"]:
                            st.session_state.thread_id = None
                            st.session_state.messages = []
                        st.rerun()

        except Exception as e:
            st.error(f"Could not load sessions: {e}")

    return st.session_state.get("thread_id"), DEFAULT_CONFIG_ID