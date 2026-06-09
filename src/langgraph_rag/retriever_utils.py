import os
import json
import logging
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

CONFIGS_PATH = "./experiments/configs_langgraph.json"
CACHE_DIR = "./data/cache"

_retriever_cache: dict[str, Retriever] = {}


def get_retriever(config: dict) -> Retriever:
    """Load or return cached Retriever for given config."""
    index_id = config["index_id"]
    if index_id not in _retriever_cache:
        cache_dir = os.path.join(CACHE_DIR, index_id)
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Index '{index_id}' not found. Run build_indexes.py first.")
        logger.info(f"Loading retriever for index: {index_id}")
        retriever = Retriever()
        retriever.build_index([], index_name=index_id)
        _retriever_cache[index_id] = retriever
    return _retriever_cache[index_id]


def load_config(config_id: str) -> dict:
    """Load config by id from configs_langgraph.json."""
    if not os.path.exists(CONFIGS_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIGS_PATH}")
    with open(CONFIGS_PATH, "r") as f:
        config_file = json.load(f)
    for config in config_file["configs"]:
        if config["id"] == config_id:
            return config
    raise ValueError(f"Config '{config_id}' not found in {CONFIGS_PATH}")