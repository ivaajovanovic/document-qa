import requests

BASE_URL = "http://127.0.0.1:8000"


def get_configs() -> list[dict]:
    response = requests.get(f"{BASE_URL}/configs")
    response.raise_for_status()
    return response.json()


def get_sessions() -> list[dict]:
    response = requests.get(f"{BASE_URL}/sessions")
    response.raise_for_status()
    return response.json()


def new_session() -> str:
    response = requests.post(f"{BASE_URL}/sessions/new")
    response.raise_for_status()
    return response.json()["thread_id"]


def delete_session(thread_id: str):
    response = requests.delete(f"{BASE_URL}/sessions/{thread_id}")
    response.raise_for_status()


def get_history(thread_id: str) -> list[dict]:
    response = requests.get(f"{BASE_URL}/sessions/{thread_id}/history")
    response.raise_for_status()
    return response.json()


def send_message(thread_id: str, message: str, config_id: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "thread_id": thread_id,
            "message": message,
            "config_id": config_id,
        }
    )
    response.raise_for_status()
    return response.json()  # {"answer": "...", "thread_id": "...", "images": [...]}