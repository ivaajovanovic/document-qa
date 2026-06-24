import uuid
from langchain_core.messages import HumanMessage, AIMessage

from src.langgraph_rag.graph import rag_graph
from src.langgraph_rag.state import RAGState

DEFAULT_CONFIG = "config_multimodal_k10_rrf60"


def create_initial_state(config_id: str = DEFAULT_CONFIG) -> RAGState:
    return {
        "messages": [],
        "config_id": config_id,
        "retrieved_chunks": None,
        "answer": None,
        "error": None,
    }


def ask_chatbot(
    message: str,
    state: RAGState | None = None,
    thread_id: str | None = None,
    config_id: str = DEFAULT_CONFIG,
) -> dict:
    thread_id = thread_id or str(uuid.uuid4())
    graph_config = {"configurable": {"thread_id": thread_id}}

    if state is None:
        state = create_initial_state(config_id)

    state["config_id"] = config_id or DEFAULT_CONFIG
    state["messages"].append(HumanMessage(content=message))

    state = rag_graph.invoke(state, config=graph_config)

    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )

    return {
        "answer": last_ai.content if last_ai else "",
        "thread_id": thread_id,
        "state": state,
    }