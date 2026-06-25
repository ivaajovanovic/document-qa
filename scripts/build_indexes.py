import sys
sys.path.append(".")

import os
import json
import shutil
import logging
from dotenv import load_dotenv
from src.rag.retriever import Retriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "./data/cache"


def load_all_chunks(processed_dir: str) -> list[dict]:
    """Load all chunks from processed directory."""
    all_chunks = []
    for filename in sorted(os.listdir(processed_dir)):
        if not filename.endswith("_chunks.json"):
            continue
        with open(os.path.join(processed_dir, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    logger.info(f"Loaded {len(all_chunks)} chunks from {processed_dir}")
    return all_chunks


def build_all_indexes() -> None:
    with open("./experiments/configs_langgraph.json", "r") as f:
        config_file = json.load(f)

    indexes = config_file["indexes"]

    # obrisi stare cache fajlove
    logger.info("Cleaning old cache...")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for index_cfg in indexes:
        idx_id = index_cfg["id"]
        processed_dir = index_cfg["processed_dir"]

        if not os.path.exists(processed_dir):
            logger.warning(f"Skipping {idx_id} — {processed_dir} not found")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Building: {idx_id} — {index_cfg['description']}")

        chunks = load_all_chunks(processed_dir)
        if not chunks:
            continue

        retriever = Retriever()
        retriever.build_index(chunks, index_name=idx_id)
        logger.info(f"Done: {idx_id} ({len(chunks)} chunks)")

    logger.info("\nAll indexes built!")


if __name__ == "__main__":
    build_all_indexes()