import os
import json
import re
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


def _get_chat_history(state: RAGState, n: int = 6) -> str:
    """Extract last n messages as formatted chat history string."""
    history_messages = state["messages"][:-1][-n:]
    history = ""
    for msg in history_messages:
        if isinstance(msg, HumanMessage):
            history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage) and msg.content:
            history += f"Assistant: {msg.content[:300]}\n"
    return history.strip()


def parse_query_node(state: RAGState) -> RAGState:
    """Decide if the last message is a /config command or a question."""
    if not state.get("config_id"):
        state["config_id"] = "config_256_k10_rrf60"

    last_message = state["messages"][-1]

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


def decompose_query_node(state: RAGState) -> RAGState:
    """
    History-aware query rewriting and decomposition:
    1. Resolve pronouns and references using chat history
    2. Decompose into sub-queries if comparative or multi-document question
    """
    last_message = state["messages"][-1]
    content = last_message.content
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    question = content.strip()

    history = _get_chat_history(state)

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        if history:
            prompt = f"""Given the conversation history and the latest question, do two things:
1. Rewrite the question to be self-contained — replace pronouns like "it", "they", "this", "that paper", "that method" with explicit references from history.
2. If the rewritten question requires multiple papers or concepts, decompose into 2-4 specific search queries. Otherwise return 1 query.

Conversation history:
{history}

Latest question: {question}

Rules:
- Keep queries short and specific (3-6 words)
- Use paper names or technical terms when possible
- For comparative questions return at least 2 queries
- Example: "Compare BERT and Transformer attention" → ["BERT bidirectional attention", "Transformer self-attention Vaswani 2017"]

Respond with ONLY a JSON array of strings:
["query1", "query2"]"""
        else:
            prompt = f"""Analyze this question about academic NLP papers.
Decompose into 2-4 specific search queries if it requires multiple papers or concepts.
Return 1 query for simple questions.

Question: {question}

Rules:
- Keep queries short and specific (3-6 words)
- Use paper names or technical terms when possible
- For comparative questions return at least 2 queries
- Example: "Compare BERT and Transformer attention" → ["BERT bidirectional attention", "Transformer self-attention Vaswani 2017"]

Respond with ONLY a JSON array of strings:
["query1", "query2"]"""

        response = llm.invoke(prompt)
        text = response.content.strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > 0:
            sub_queries = json.loads(text[start:end])
            sub_queries = [q for q in sub_queries if isinstance(q, str) and len(q) > 2]
            if sub_queries:
                state["sub_queries"] = sub_queries
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                return state

    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}, using original query")

    state["sub_queries"] = [question]
    logger.info(f"Using original query: {question}")
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
    max_retries = 3

    for attempt in range(max_retries):
        try:
            agent = get_agent()

            system = SystemMessage(content="""You are a helpful research assistant for academic papers about NLP and deep learning.
ALWAYS use search_papers tool to answer questions about paper content.

Tool selection guide:
- COMPANIES or INSTITUTIONS → use search_papers_by_company
- PERSON NAMES → use search_papers_by_author
- YEARS or TIME PERIODS → use search_papers_by_year or list_papers_metadata
- GENERAL CONTENT → use search_papers
- AUTHORS OF A PAPER → use get_paper_authors

Do NOT use tools for greetings.""")

            messages = [system] + state["messages"]

            sub_queries = state.get("sub_queries", [])
            if len(sub_queries) > 1:
                hint = f"Please search for each of these topics separately: {', '.join(sub_queries)}"
                messages = messages + [HumanMessage(content=hint)]

            response = agent.invoke(messages)
            state["messages"] = [response]
            state["error"] = None
            return state

        except Exception as e:
            logger.warning(f"Agent attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Agent failed after {max_retries} attempts, falling back to direct search")
                from src.langgraph_rag.tools import search_papers
                from langchain_core.messages import ToolMessage

                question = next(
                    (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                    ""
                )
                try:
                    result = search_papers.invoke({"query": question})
                    state["messages"].append(
                        ToolMessage(content=result, tool_call_id="fallback_search")
                    )
                    state["error"] = None
                except Exception as fallback_e:
                    logger.error(f"Fallback search also failed: {fallback_e}")
                    state["messages"] = [AIMessage(content="I encountered an error. Please try again.")]
                    state["error"] = str(fallback_e)

    return state


def tools_generate_node(state: RAGState) -> RAGState:
    try:
        tool_results = []
        question = ""
        raw_chunks = []

        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_results.append(msg.content)
                raw_chunks.append({
                    "text": msg.content,
                    "score": None,
                    "metadata": {"title": "Tool Result"}
                })
            elif isinstance(msg, HumanMessage):
                question = msg.content
                break

        state["retrieved_chunks"] = {"hybrid": raw_chunks}

        if not tool_results:
            state["messages"] = [AIMessage(content="I could not find sufficient evidence in this paper collection to answer this question.")]
            return state

        context = "\n\n".join(tool_results)
        history = _get_chat_history(state)

        # grupišu sources po paperu
        sources_dict = {}
        for result in tool_results:
            for line in result.split("\n"):
                match = re.match(r'\[(.+?) \((\d{4}\.\d+)\), page (\d+)\]', line)
                if match:
                    title, arxiv_id, page = match.group(1), match.group(2), match.group(3)
                    if arxiv_id not in sources_dict:
                        sources_dict[arxiv_id] = {"title": title, "pages": []}
                    if page not in sources_dict[arxiv_id]["pages"]:
                        sources_dict[arxiv_id]["pages"].append(page)

        # formatiraj kao "Title (arxiv_id), pages: 1, 3, 5"
        sources = []
        for arxiv_id, info in sources_dict.items():
            pages = ", ".join(sorted(info["pages"], key=int))
            sources.append(f"{info['title']} ({arxiv_id}), pages: {pages}")

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        if history:
            prompt = f"""You are a research assistant specializing in NLP and deep learning papers.
Answer ONLY based on the retrieved context below. Do NOT use any external knowledge or training data.

Conversation history:
{history}

Retrieved context from academic papers:
{context}

Current question: {question}

Rules:
- Use ONLY information explicitly present in the retrieved context above
- Cite specific details: method names, numbers, findings, page references
- For comparison questions, explicitly contrast the two approaches point by point
- If the retrieved context does not contain sufficient evidence to answer the question, respond with:
  "I could not find sufficient evidence in this paper collection to answer this question."
- Do NOT add sources at the end — they will be added automatically
- Do NOT speculate or add information not present in the context"""
        else:
            prompt = f"""You are a research assistant specializing in NLP and deep learning papers.
Answer ONLY based on the retrieved context below. Do NOT use any external knowledge or training data.

Retrieved context from academic papers:
{context}

Question: {question}

Rules:
- Use ONLY information explicitly present in the retrieved context above
- Cite specific details: method names, numbers, findings, page references
- For comparison questions, explicitly contrast the two approaches point by point
- If the retrieved context does not contain sufficient evidence to answer the question, respond with:
  "I could not find sufficient evidence in this paper collection to answer this question."
- Do NOT add sources at the end — they will be added automatically
- Do NOT speculate or add information not present in the context"""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        # dodaj sources samo ako odgovor nije "not found"
        not_found_phrases = [
            "could not find sufficient evidence",
            "does not contain enough information",
            "not possible to provide",
            "context does not contain",
        ]
        answer_lower = answer.lower()
        should_add_sources = sources and not any(phrase in answer_lower for phrase in not_found_phrases)

        if should_add_sources:
            sources_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
            answer = answer + sources_text

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


def execute_sub_queries_node(state: RAGState) -> RAGState:
    """
    Directly execute search_papers for each sub_query without going through agent.
    Bypasses LLM tool calling to avoid format bugs with complex queries.
    """
    from src.langgraph_rag.tools import search_papers
    from langchain_core.messages import ToolMessage

    sub_queries = state.get("sub_queries", [])
    logger.info(f"Executing {len(sub_queries)} sub-queries directly")

    for i, query in enumerate(sub_queries):
        try:
            result = search_papers.invoke({"query": query})
            state["messages"].append(
                ToolMessage(content=result, tool_call_id=f"sub_query_{i}")
            )
            logger.info(f"Sub-query {i+1}/{len(sub_queries)}: {query}")
        except Exception as e:
            logger.error(f"Sub-query failed: {query} — {e}")

    return state