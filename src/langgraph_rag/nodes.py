import os
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from src.langgraph_rag.tools import TOOLS
from src.rag.generator import generate_answer
from src.langgraph_rag.state import RAGState
from src.langgraph_rag.retriever_utils import get_retriever, load_config

logger = logging.getLogger(__name__)

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
        _agent = llm.bind_tools(TOOLS)
    return _agent


def parse_query_node(state: RAGState) -> RAGState:
    """Decide if the last message is a /config command or a question."""

    if not state.get("config_id"):
        state["config_id"] = "config_256_k10_rrf60"

    last_message = state["messages"][-1]

    # handle both string and list content
    content = last_message.content
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    content = content.strip()

    if content.startswith("/config"):
        parts = content.split()
        if len(parts) == 2:
            state["config_id"] = parts[1]
            logger.info(f"Config updated to: {parts[1]}")
        else:
            state["error"] = "Usage: /config <config_id>"

    return state


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks for the last question using active config."""
    last_message = state["messages"][-1]
    query = last_message.content.strip()

    try:
        config = load_config(state["config_id"])
        retriever = get_retriever(config)
        search_results = retriever.search(query, top_k=config["top_k"], use_rrf=True)
        state["retrieved_chunks"] = search_results
        state["error"] = None
        logger.info(f"Retrieved {len(search_results['hybrid'])} chunks")

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["error"] = str(e)
        state["retrieved_chunks"] = None

    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate answer based on retrieved chunks."""
    if state.get("error") or not state.get("retrieved_chunks"):
        state["messages"] = [AIMessage(content=f"Error: {state.get('error', 'No chunks retrieved')}")]
        return state

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )
    question = last_human.content.strip() if last_human else ""

    try:
        hybrid_chunks = state["retrieved_chunks"]["hybrid"]
        answer = generate_answer(question, hybrid_chunks)
        state["answer"] = answer
        state["messages"] = [AIMessage(content=answer)]

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state["messages"] = [AIMessage(content=f"Error generating answer: {e}")]

    return state


def agent_node(state: RAGState) -> RAGState:
    """Groq agent node — LLM decides which tool to call."""
    try:
        agent = get_agent()
        system = SystemMessage(content="""You are a helpful research assistant for academic papers about NLP and deep learning.
Use the provided tools to answer questions about academic papers.
Use tools when:
- filtering by year, author, or company
- searching for specific paper content
Do NOT use tools for greetings or general conversation - just respond directly.""")
        messages = [system] + state["messages"]
        response = agent.invoke(messages)
        state["messages"] = [response]
        state["error"] = None
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        state["messages"] = [AIMessage(content="I encountered an error while processing your request. Please try again or rephrase your question.")]
        state["error"] = str(e)
    return state


def tools_generate_node(state: RAGState) -> RAGState:
    """Generate final answer after tool execution."""
    try:
        tool_results = []
        question = ""

        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_results.append(msg.content)
            elif isinstance(msg, HumanMessage):
                question = msg.content
                break

        if not tool_results:
            state["messages"] = [AIMessage(content="No results found from tools.")]
            return state

        context = "\n\n".join(tool_results)
        
        # koristi Groq za bolji odgovor
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        prompt = f"""Based on the following context from academic papers, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
        response = llm.invoke(prompt)
        answer = response.content
        state["answer"] = answer
        state["messages"] = [AIMessage(content=answer)]

    except Exception as e:
        logger.error(f"Tools generate failed: {e}")
        state["messages"] = [AIMessage(content="I encountered an error while generating the answer. Please try again.")]
        state["error"] = str(e)

    return state


def update_config_node(state: RAGState) -> RAGState:
    """Confirm config update and add message to conversation."""
    if state.get("error"):
        state["messages"] = [AIMessage(content=f"Error: {state['error']}")]
    else:
        config_id = state["config_id"]
        try:
            config = load_config(config_id)
            msg = (
                f"Config updated to: **{config_id}**\n"
                f"- chunk_size: {config['chunk_size']}\n"
                f"- top_k: {config['top_k']}\n"
                f"- rrf_k: {config['rrf_k']}\n"
                f"- context_offset: {config['context_offset']}\n"
                f"- {config['description']}"
            )
        except ValueError as e:
            msg = str(e)
        state["messages"] = [AIMessage(content=msg)]

    return state