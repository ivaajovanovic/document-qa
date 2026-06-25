import os
import json
import logging
from src.rag.retriever import Retriever as TextRetriever
from src.rag.retriever_multimodal import Retriever as MultimodalRetriever

logger = logging.getLogger(__name__)

CONFIGS_PATH = "./experiments/configs_langgraph.json"
CACHE_DIR = "./data/cache"

_retriever_cache: dict[str, TextRetriever] = {}


def get_retriever(config: dict) -> TextRetriever:
    """Return a retriever instance for the selected index configuration.

    Retrievers are cached in memory so repeated queries do not reload FAISS/BM25
    from disk on every request.
    """
    index_id = config["index_id"]
    retriever_type = config.get("retriever_type", "text")

    if index_id not in _retriever_cache:
        cache_dir = os.path.join(CACHE_DIR, index_id)
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Index '{index_id}' not found. Run build_indexes.py first.")

        if retriever_type == "multimodal":
            logger.info(f"Loading multimodal retriever for index: {index_id}")
            retriever = MultimodalRetriever()
        else:
            logger.info(f"Loading text retriever for index: {index_id}")
            retriever = TextRetriever()

        # Passing empty chunks here is intentional: `build_index` loads existing
        # cached FAISS/BM25 index files when `index_name` already exists.
        retriever.build_index([], index_name=index_id)
        _retriever_cache[index_id] = retriever
    return _retriever_cache[index_id]


def load_config(config_id: str) -> dict:
    """Read one retrieval config by id from `configs_langgraph.json`."""
    if not os.path.exists(CONFIGS_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIGS_PATH}")
    with open(CONFIGS_PATH, "r") as f:
        config_file = json.load(f)
    for config in config_file["configs"]:
        if config["id"] == config_id:
            return config
    raise ValueError(f"Config '{config_id}' not found in {CONFIGS_PATH}")