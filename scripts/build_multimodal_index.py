import sys
sys.path.append(".")

import os
import json
import logging
from dotenv import load_dotenv
from src.rag.retriever import Retriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = "./data/processed_multimodal"
INDEX_NAME = "idx_multimodal"


def load_multimodal_chunks(processed_dir: str) -> list[dict]:
    chunks = []
    for fname in os.listdir(processed_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                chunks.extend(data)
    return chunks


if __name__ == "__main__":
    print(f"Loading multimodal chunks from {PROCESSED_DIR}...")
    chunks = load_multimodal_chunks(PROCESSED_DIR)

    text_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "text"]
    fig_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "figure"]

    print(f"Total chunks: {len(chunks)}")
    print(f"  Text chunks: {len(text_chunks)}")
    print(f"  Figure chunks: {len(fig_chunks)}")

    print(f"\nBuilding index '{INDEX_NAME}'...")
    retriever = Retriever()
    retriever.build_index(chunks, index_name=INDEX_NAME)

    print(f"\nDone! Index saved to ./data/cache/{INDEX_NAME}")
    print("Test search:")
    results = retriever.search("transformer attention mechanism", top_k=3)
    for r in results["hybrid"]:
        chunk_type = r["metadata"].get("chunk_type", "text")
        title = r["metadata"].get("title", "")
        print(f"  [{chunk_type}] {title} — {r['text'][:80]}")