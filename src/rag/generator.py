import os
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a helpful research assistant specializing in academic papers.
Your answers must be grounded ONLY in the provided context from the paper collection.
If the answer cannot be found in the context, say "I cannot find this information in the provided papers."
Do not use any knowledge outside of the provided context.
Always cite which paper the information comes from."""


def get_generator() -> ChatOllama:
    """Returns ChatOllama instance."""
    return ChatOllama(model=GENERATOR_MODEL)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generate answer to question based on retrieved context chunks.

    Args:
        question: User question.
        context_chunks: List of retrieved chunks with text and metadata.

    Returns:
        Generated answer string.
    """
    context = "\n\n".join(
        f"[Paper: {chunk['metadata'].get('title', 'Unknown')}]\n{chunk['text']}"
        for chunk in context_chunks
    )

    user_message = f"""Context from academic papers:
{context}

Question: {question}"""

    try:
        llm = get_generator()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return ""