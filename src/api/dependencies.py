"""Shared dependencies for API endpoints.

Loads configuration definitions used across endpoints for retrieval setup.
"""

import json


def get_configs() -> list[dict]:
    """Load all available retrieval configurations.
    
    Reads from `experiments/configs_langgraph.json` and returns the list of
    retrieval configs that can be selected by users (chunk size, top_k, rrf_k, etc.).
    
    Returns:
        List of configuration dictionaries, each with keys: id, description, chunk_size, 
        top_k, rrf_k, index_id, retriever_type.
    """
    with open("./experiments/configs_langgraph.json") as f:
        return json.load(f)["configs"]