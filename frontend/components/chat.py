"""Chat message rendering and user input handling.

Manages the main chat interface: message history display, user input,
RAG query submission, answer display, and retrieved figure visualization.
"""

import streamlit as st
from ..api_client import send_message


def render_chat(thread_id: str, config_id: str):
    """Render chat interface with message history and input box.
    
    Args:
        thread_id: Session ID for sending messages.
        config_id: Retrieval config ID to use for this session.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not thread_id:
        st.markdown("""
        <div style="text-align:center; padding:100px 20px;">
            <div style="font-size:3rem;">🔮</div>
            <h2 style="color:#c4b5fd; margin-top:12px; font-weight:600;">Document QA</h2>
            <p style="color:#6b7280; font-size:0.9rem; margin-top:8px;">
                Ask questions about your document collection.<br>
                Start a new chat from the sidebar.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("images"):
                _render_images(msg["images"])

    if prompt := st.chat_input("Ask about the papers..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "images": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    result = send_message(thread_id, prompt, config_id)
                    answer = result.get("answer", "")
                    images = result.get("images", [])

                    st.markdown(answer)
                    if images:
                        _render_images(images)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "images": images,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")


def _render_images(images: list[str]):
    """Display retrieved figure images in a compact grid layout.
    
    Args:
        images: List of base64-encoded image data URLs.
    """
    if not images:
        return
    st.markdown(
        '<p style="font-size:0.75rem; color:#6b7280; margin-top:10px; margin-bottom:4px;">'
        '📎 Retrieved figures</p>',
        unsafe_allow_html=True
    )
    cols = st.columns(min(len(images), 3))
    for i, img_data in enumerate(images[:3]):
        with cols[i]:
            st.image(img_data, width=200)