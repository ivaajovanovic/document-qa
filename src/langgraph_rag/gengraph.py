import os
import sys
sys.path.insert(0, ".")

from src.langgraph_rag.graph import build_graph

os.makedirs("results", exist_ok=True)
graph = build_graph()

with open("results/graph.mmd", "w") as f:
    f.write(graph.get_graph().draw_mermaid())

print("Saved!")