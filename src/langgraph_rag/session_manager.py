import json
import os
from datetime import datetime

SESSIONS_FILE = "./data/cache/sessions.json"


def _load_sessions() -> dict:
    """Load session metadata JSON from disk (or return empty structure)."""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_sessions(sessions: dict):
    """Persist session metadata JSON to disk."""
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def create_session(thread_id: str, first_question: str):
    """Create new session record and derive short display name from first prompt."""
    sessions = _load_sessions()
    name = " ".join(first_question.split()[:5])
    sessions[thread_id] = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "message_images": {},  # message_index -> [image_paths]
    }
    _save_sessions(sessions)


def update_session(thread_id: str):
    """Refresh `last_updated` timestamp for an existing session."""
    sessions = _load_sessions()
    if thread_id in sessions:
        sessions[thread_id]["last_updated"] = datetime.now().isoformat()
        _save_sessions(sessions)


def save_message_images(thread_id: str, message_index: int, image_paths: list[str]):
    """Save image paths for a specific message index."""
    sessions = _load_sessions()
    if thread_id not in sessions:
        return
    if "message_images" not in sessions[thread_id]:
        sessions[thread_id]["message_images"] = {}
    sessions[thread_id]["message_images"][str(message_index)] = image_paths
    _save_sessions(sessions)


def get_message_images(thread_id: str) -> dict:
    """Get all message image paths for a session. Returns {message_index: [paths]}."""
    sessions = _load_sessions()
    if thread_id not in sessions:
        return {}
    return sessions[thread_id].get("message_images", {})


def get_all_sessions() -> list[dict]:
    """Return sessions sorted by most recently updated first."""
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
    """Remove one session record from storage if present."""
    sessions = _load_sessions()
    if thread_id in sessions:
        del sessions[thread_id]
        _save_sessions(sessions)