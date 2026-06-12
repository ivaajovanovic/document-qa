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
    tools_generate_node,
)
from src.langgraph_rag.tools import TOOLS

logger = logging.getLogger(__name__)


def route_after_parse(state: RAGState) -> str:
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
    sub_queries = state.get("sub_queries", [])
    if len(sub_queries) > 1:
        return "execute_sub_queries"
    return "agent"


def route_after_agent(state: RAGState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "tools_generate"


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState, input=InputState)

    graph.add_node("parse_query", parse_query_node)
    graph.add_node("decompose_query", decompose_query_node)
    graph.add_node("execute_sub_queries", execute_sub_queries_node)
    graph.add_node("update_config", update_config_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("tools_generate", tools_generate_node)

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

    graph.add_edge("execute_sub_queries", "tools_generate")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "tools_generate": "tools_generate",
        }
    )
    graph.add_edge("tools", "tools_generate")
    graph.add_edge("tools_generate", END)

    checkpointer = SqliteSaver.from_conn_string("./data/cache/chat_memory.db")
    return graph.compile(checkpointer=checkpointer)


rag_graph = build_graph()