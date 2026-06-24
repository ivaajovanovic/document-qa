import streamlit as st
from components.sidebar import render_sidebar
from components.chat import render_chat

st.set_page_config(
    page_title="Document QA",
    page_icon="🔮",
    layout="wide",
)

st.markdown("""
<style>
/* glavna pozadina */
.stApp {
    background-color: #0f0f1a;
}

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
    border-right: 1px solid #7c3aed;
}

/* sidebar title */
[data-testid="stSidebar"] h1 {
    color: #a78bfa;
    font-size: 1.4rem;
}

/* sidebar subheader */
[data-testid="stSidebar"] h2 {
    color: #7c3aed;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* dugmad u sidebaru */
[data-testid="stSidebar"] .stButton > button {
    background-color: #2d1b69;
    color: #e2d9f3;
    border: 1px solid #7c3aed;
    border-radius: 8px;
    transition: all 0.2s;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #7c3aed;
    color: white;
    border-color: #a78bfa;
}

/* new chat dugme */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background-color: #7c3aed;
    color: white;
    font-weight: 600;
    border: none;
}

[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background-color: #6d28d9;
}

/* chat poruke */
[data-testid="stChatMessage"] {
    background-color: #1a1a2e;
    border-radius: 12px;
    border: 1px solid #2d1b69;
    margin-bottom: 8px;
}

/* chat input */
[data-testid="stChatInput"] {
    background-color: #1a1a2e;
    border: 1px solid #7c3aed;
    border-radius: 12px;
}

/* send button */
[data-testid="stChatInput"] button {
    background-color: #7c3aed !important;
    border-radius: 8px !important;
    color: white !important;
}

/* divider */
hr {
    border-color: #2d1b69;
}

/* scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #0f0f1a;
}
::-webkit-scrollbar-thumb {
    background: #7c3aed;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

thread_id, config_id = render_sidebar()
render_chat(thread_id, config_id)