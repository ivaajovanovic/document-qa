import json
import os
from datetime import datetime

SESSIONS_FILE = "./data/cache/sessions.json"


def _load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_sessions(sessions: dict):
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def create_session(thread_id: str, first_question: str):
    """Create new session with auto-generated name from first question."""
    sessions = _load_sessions()
    name = " ".join(first_question.split()[:5])
    sessions[thread_id] = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }
    _save_sessions(sessions)


def update_session(thread_id: str):
    """Update last_updated timestamp."""
    sessions = _load_sessions()
    if thread_id in sessions:
        sessions[thread_id]["last_updated"] = datetime.now().isoformat()
        _save_sessions(sessions)


def get_all_sessions() -> list[dict]:
    """Return all sessions sorted by last_updated descending."""
    sessions = _load_sessions()
    result = []
    for thread_id, meta in sessions.items():
        result.append({
            "thread_id": thread_id,
            "name": meta["name"],
            "created_at": meta["created_at"],
            "last_updated": meta["last_updated"],
        })
    return sorted(result, key=lambda x: x["last_updated"], reverse=True)


def delete_session(thread_id: str):
    """Delete session metadata (LangGraph checkpoint ostaje u SQLite)."""
    sessions = _load_sessions()
    if thread_id in sessions:
        del sessions[thread_id]
        _save_sessions(sessions)