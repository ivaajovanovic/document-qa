"""API client for communicating with the Document QA backend.

Provides functions to:
- Send chat queries and receive RAG answers
- Manage sessions (create, list, delete)
- Retrieve conversation history
- Load retrieval configurations
"""

import os
import requests
from src.api.schemas import ChatResponse, NewSessionResponse

BASE_URL = os.environ.get("DOCUMENT_QA_API", "http://127.0.0.1:8000")


def _request(method: str, path: str, **kwargs):
    """Make HTTP request to backend API.
    
    Args:
        method: HTTP method (GET, POST, DELETE).
        path: API endpoint path (e.g., '/chat').
        **kwargs: Additional arguments to pass to requests.request().
        
    Returns:
        Parsed JSON response or raw text if JSON parsing fails.
        
    Raises:
        requests.HTTPError: If HTTP status indicates error (4xx, 5xx).
    """
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, **kwargs)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return resp.text


def get_configs() -> list[dict]:
    """Fetch all available retrieval configurations.
    
    Returns:
        List of config dicts with 'id' and 'description' for user selection.
    """
    return _request("GET", "/configs")


def get_sessions() -> list[dict]:
    """List all chat sessions ordered by most recent first.
    
    Returns:
        List of session metadata dicts with thread_id, name, created_at, last_updated.
    """
    return _request("GET", "/sessions")


def new_session() -> str:
    """Create a new chat session.
    
    Returns:
        Thread ID for the new session.
    """
    data = _request("POST", "/sessions/new")
    try:
        parsed = NewSessionResponse.model_validate(data)
        return parsed.thread_id
    except Exception:
        # fallback: try to read raw
        return data.get("thread_id") if isinstance(data, dict) else data


def delete_session(thread_id: str):
    """Delete a chat session and its conversation history.
    
    Args:
        thread_id: Session ID to delete.
    """
    return _request("DELETE", f"/sessions/{thread_id}")


def get_history(thread_id: str) -> list[dict]:
    """Retrieve full conversation history for a session.
    
    Args:
        thread_id: Session ID.
        
    Returns:
        List of message dicts with role, content, and images.
    """
    return _request("GET", f"/sessions/{thread_id}/history")


def send_message(thread_id: str, message: str, config_id: str) -> dict:
    """Send a user message to the RAG pipeline and get answer.
    
    Args:
        thread_id: Session ID for conversation persistence.
        message: User query text.
        config_id: Retrieval configuration ID.
        
    Returns:
        Dict with 'answer' (generated response) and 'images' (base64 figures).
    """
    data = _request(
        "POST",
        "/chat",
        json={
            "thread_id": thread_id,
            "message": message,
            "config_id": config_id,
        },
    )
    try:
        parsed = ChatResponse.model_validate(data)
        return parsed.model_dump()
    except Exception:
        return data