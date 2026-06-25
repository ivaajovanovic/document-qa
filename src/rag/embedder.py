import os
import logging
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

"""Embedding helpers for an initial RAG prototype.

The goal here is to keep embedding setup simple so retrieval experiments are
easy to run and reason about.
"""

load_dotenv()

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")


def get_embeddings() -> OllamaEmbeddings:
    """
    Create the embedding model object used across retriever code.

    Keeping this in one place makes model swaps easy while experimenting.
    """
    return OllamaEmbeddings(model=EMBED_MODEL)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Attach vector embeddings to chunk dictionaries.

    Args:
        chunks: List of chunk dicts with "text" field.

    Returns:
        New list where each chunk includes an ``embedding`` field.
    """
    embeddings = get_embeddings()
    texts = [chunk["text"] for chunk in chunks]
    
    logger.info(f"Embedding {len(texts)} chunks...")
    vectors = embeddings.embed_documents(texts)
    
    embedded = []
    for chunk, vector in zip(chunks, vectors):
        embedded.append({**chunk, "embedding": vector})
    
    logger.info(f"Done - embedded {len(embedded)} chunks")
    return embedded