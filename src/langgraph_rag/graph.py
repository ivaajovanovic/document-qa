import sqlite3
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from src.langgraph_rag.state import InputState, RAGState
from src.langgraph_rag.nodes import (
    parse_query_node,
    decompose_query_node,
    execute_sub_queries_node,
    update_config_node,
    agent_node,
    direct_answer_node,
    tools_generate_node,
    METADATA_TOOLS,
)
from src.langgraph_rag.tools import TOOLS

logger = logging.getLogger(__name__)


def route_after_parse(state: RAGState) -> str:
    """Choose next node after parsing user input.

    `/config ...` commands go to config update node.
    All normal questions continue to decomposition.
    """
    last_message = state["messages"][-1]
    content = last_message.content
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    content = content.strip().lower()
    if content.startswith("/config"):
        return "update_config"
    return "decompose_query"


def route_after_decompose(state: RAGState) -> str:
    """Route to multi-query execution only when decomposition produced >1 query."""
    sub_queries = state.get("sub_queries", [])
    if len(sub_queries) > 1:
        return "execute_sub_queries"
    return "agent"


def route_after_agent(state: RAGState) -> str:
    """If agent requested tools, go to tool node; otherwise generate answer directly."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "generate_answer"


def route_after_tools(state: RAGState) -> str:
    """Decide how to respond after tool execution.

    Metadata-only tools already return final text, so we skip LLM synthesis and
    return the tool output directly. Retrieval tools continue to answer generation.
    """
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "")
            for prev_msg in reversed(state["messages"]):
                if hasattr(prev_msg, "tool_calls") and prev_msg.tool_calls:
                    for tc in prev_msg.tool_calls:
                        if tc.get("id") == tool_call_id:
                            tool_name = tc.get("name", "")
                            print(f"[ROUTE_TOOLS] tool_name={tool_name}")
                            if tool_name in METADATA_TOOLS:
                                return "direct_answer"
                            return "generate_answer"
            break
    print(f"[ROUTE_TOOLS] no tool found → generate_answer")
    return "generate_answer"


def build_graph() -> StateGraph:
    """Construct and compile the end-to-end LangGraph workflow.

    The graph uses SQLite checkpointing so conversation state survives between
    requests for the same `thread_id`.
    """
    graph = StateGraph(RAGState, input=InputState)

    graph.add_node("parse_query", parse_query_node)
    graph.add_node("decompose_query", decompose_query_node)
    graph.add_node("execute_sub_queries", execute_sub_queries_node)
    graph.add_node("update_config", update_config_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("generate_answer", tools_generate_node)

    graph.add_edge(START, "parse_query")

    graph.add_conditional_edges(
        "parse_query",
        route_after_parse,
        {
            "update_config": "update_config",
            "decompose_query": "decompose_query",
        }
    )

    graph.add_edge("update_config", END)

    graph.add_conditional_edges(
        "decompose_query",
        route_after_decompose,
        {
            "execute_sub_queries": "execute_sub_queries",
            "agent": "agent",
        }
    )

    graph.add_edge("execute_sub_queries", "generate_answer")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "generate_answer": "generate_answer",
        }
    )

    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "direct_answer": "direct_answer",
            "generate_answer": "generate_answer",
        }
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("generate_answer", END)

    import os
    os.makedirs("./data/cache", exist_ok=True)
    # Shared SQLite file stores LangGraph checkpoints by thread id.
    conn = sqlite3.connect("./data/cache/chat_memory.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


rag_graph = build_graph()