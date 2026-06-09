import os
import logging
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")


def get_embeddings() -> OllamaEmbeddings:
    """
    Returns OllamaEmbeddings instance.
    """
    return OllamaEmbeddings(model=EMBED_MODEL)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for all chunks.

    Args:
        chunks: List of chunk dicts with "text" field.

    Returns:
        Same chunks with "embedding" field added.
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