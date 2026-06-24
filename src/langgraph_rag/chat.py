import json
import logging
from dotenv import load_dotenv

from src.langgraph_rag.chat_service import (
    DEFAULT_CONFIG,
    create_initial_state,
    ask_chatbot,
)

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_help_text() -> str:
    with open("./experiments/configs_langgraph.json", encoding="utf-8") as f:
        configs = json.load(f)["configs"]

    config_lines = "\n".join(
        f"  {c['id']}  — {c['description']}"
        for c in configs
    )

    return f"""
=== RAG Chat ===
Commands:
  /config <config_id>  — change retrieval configuration
  /configs             — list available configurations
  /chunks              — show retrieved chunks from last query
  /help                — show this message
  /exit                — exit chat

Available configs:
{config_lines}
"""


def print_chunks(state: dict) -> None:
    retrieved = state.get("retrieved_chunks")

    if not retrieved:
        print("No chunks retrieved yet.")
        return

    chunks = retrieved.get("hybrid", [])

    if not chunks:
        print("No chunks retrieved yet.")
        return

    print(f"\nLast retrieved chunks ({len(chunks)}):")

    for i, c in enumerate(chunks, start=1):
        metadata = c.get("metadata", {})
        title = metadata.get("title") or metadata.get("paper_title") or "Unknown"
        page = metadata.get("page_num") or metadata.get("page")
        arxiv_id = metadata.get("arxiv_id")
        score = c.get("score")
        text = c.get("text") or c.get("page_content") or ""

        print(
            f"\n[{i}] score={score} | {title} | "
            f"arxiv={arxiv_id} | page={page}"
        )
        print(text[:1200])
        print("-" * 80)


def run_chat() -> None:
    """Run interactive RAG chat in terminal."""

    print(get_help_text())
    print(f"Active config: {DEFAULT_CONFIG}")
    print("=" * 40)

    state = create_initial_state(DEFAULT_CONFIG)
    thread_id = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered == "/exit":
            print("Goodbye!")
            break

        if lowered == "/help":
            print(get_help_text())
            continue

        if lowered == "/configs":
            with open("./experiments/configs_langgraph.json", encoding="utf-8") as f:
                configs = json.load(f)["configs"]

            for c in configs:
                marker = "← active" if c["id"] == state.get("config_id") else ""
                print(f"  {c['id']} — {c['description']} {marker}")
            continue

        if lowered == "/chunks":
            print_chunks(state)
            continue

        try:
            result = ask_chatbot(
                message=user_input,
                state=state,
                thread_id=thread_id,
                config_id=state.get("config_id", DEFAULT_CONFIG),
            )

            state = result["state"]
            thread_id = result["thread_id"]

        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Chat invocation failed")
            continue

        answer = result.get("answer") or "No answer generated."
        print(f"\nAssistant: {answer}")

        if user_input.startswith("/config"):
            print(f"\nActive config: {state.get('config_id')}")


if __name__ == "__main__":
    run_chat()