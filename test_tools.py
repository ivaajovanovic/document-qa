import sys
sys.path.append(".")
from src.rag.retriever import Retriever
from src.langgraph_rag.retriever_utils import load_config

config = load_config("config_multimodal_k10_rrf60")
retriever = Retriever()
retriever.build_index([], index_name="idx_multimodal")

# Pretraži samo figure
results = retriever.search("BERT architecture figure diagram", top_k=10)
for r in results["hybrid"]:
    chunk_type = r["metadata"].get("chunk_type", "text")
    print(f"[{chunk_type}] {r['text'][:80]}")