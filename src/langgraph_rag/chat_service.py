import uuid
from langchain_core.messages import HumanMessage, AIMessage

from src.langgraph_rag.graph import rag_graph
from src.langgraph_rag.state import RAGState

DEFAULT_CONFIG = "config_multimodal_k10_rrf60"


def create_initial_state(config_id: str = DEFAULT_CONFIG) -> RAGState:
    """Create an empty state object used by the LangGraph pipeline.

    The state keeps everything needed across turns: chat messages,
    selected retrieval config, retrieved chunks, generated answer, and error.
    """
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
    """Run one chat turn through the graph and return updated conversation state.

    Steps:
    1) ensure a thread id exists,
    2) append current user message,
    3) invoke graph,
    4) extract the latest assistant response.
    """
    # Thread id is used by LangGraph checkpointing to persist conversation memory.
    thread_id = thread_id or str(uuid.uuid4())
    graph_config = {"configurable": {"thread_id": thread_id}}

    if state is None:
        state = create_initial_state(config_id)

    state["config_id"] = config_id or DEFAULT_CONFIG
    state["messages"].append(HumanMessage(content=message))

    # Graph mutates/extends state with retrieval data, tool calls and final answer.
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