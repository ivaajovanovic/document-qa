import sys
sys.path.append(".")

import uuid
import json
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.langgraph_rag.graph import rag_graph
from src.langgraph_rag.state import RAGState

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config_256_k10_rrf60"


def get_help_text() -> str:
    with open("./experiments/configs_langgraph.json") as f:
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


def run_chat() -> None:
    """Run interactive RAG chat in terminal."""

    print(get_help_text())
    print(f"Active config: {DEFAULT_CONFIG}")
    print("="*40)

    thread_id = str(uuid.uuid4())
    graph_config = {"configurable": {"thread_id": thread_id}}

    state: RAGState = {
        "messages": [],
        "config_id": DEFAULT_CONFIG,
        "retrieved_chunks": None,
        "answer": None,
        "error": None,
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print(get_help_text())
            continue

        if user_input.lower() == "/configs":
            with open("./experiments/configs_langgraph.json") as f:
                configs = json.load(f)["configs"]
            for c in configs:
                marker = "← active" if c["id"] == state["config_id"] else ""
                print(f"  {c['id']} — {c['description']} {marker}")
            continue

        if user_input.lower() == "/chunks":
            if not state.get("retrieved_chunks"):
                print("No chunks retrieved yet.")
            else:
                chunks = state["retrieved_chunks"]["hybrid"]
                print(f"\nLast retrieved chunks ({len(chunks)}):")
                for i, c in enumerate(chunks):
                    print(f"\n[{i+1}] score={c['score']} | {c['metadata'].get('title')} | page {c['metadata'].get('page_num')}")
                    print(f"     {c['text'][:100]}...")
            continue

        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = rag_graph.invoke(state, config=graph_config)
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Graph invocation failed: {e}")
            continue

        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None
        )

        if last_ai:
            print(f"\nAssistant: {last_ai.content}")

        if user_input.startswith("/config"):
            print(f"\nActive config: {state['config_id']}")


if __name__ == "__main__":
    run_chat()