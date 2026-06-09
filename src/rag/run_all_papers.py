import sys
sys.path.append(".")

import os
import json
import logging
from dotenv import load_dotenv
from src.rag.retriever import Retriever
from src.rag.generator import generate_answer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "experiment_id": "exp_all_256_k10_rrf60",
    "chunk_size": 256,
    "top_k": 10,
    "rrf_k": 60,
    "context_offset": 150
}

PROCESSED_DIR = f"./data/processed_{CONFIG['chunk_size']}"
QA_DIR = "./data/rag_eval"
RESULTS_DIR = f"./results/{CONFIG['experiment_id']}"


def load_chunks(arxiv_id: str) -> list[dict]:
    path = os.path.join(PROCESSED_DIR, f"{arxiv_id}_chunks.json")
    if not os.path.exists(path):
        logger.warning(f"No chunks for {arxiv_id}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qa(arxiv_id: str) -> list[dict]:
    path = os.path.join(QA_DIR, f"{arxiv_id}_qa.json")
    if not os.path.exists(path):
        logger.warning(f"No QA for {arxiv_id}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    return []


def build_full_text(chunks: list[dict]) -> dict:
    full_texts = {}
    for chunk in chunks:
        arxiv_id = chunk["metadata"].get("arxiv_id")
        if arxiv_id not in full_texts:
            full_texts[arxiv_id] = ""
        full_texts[arxiv_id] += chunk["text"]
    return full_texts


def expand_context(chunk: dict, full_texts: dict, offset: int) -> str:
    arxiv_id = chunk["metadata"].get("arxiv_id")
    char_start = chunk["metadata"].get("char_start", 0)
    char_end = chunk["metadata"].get("char_end", len(chunk["text"]))
    full_text = full_texts.get(arxiv_id, chunk["text"])
    return full_text[max(0, char_start - offset):min(len(full_text), char_end + offset)]


def format_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "text": c["text"],
            "score": c.get("score"),
            "metadata": c.get("metadata", {}),
            "title": c["metadata"].get("title"),
            "arxiv_id": c["metadata"].get("arxiv_id"),
            "page_num": c["metadata"].get("page_num"),
        }
        for c in chunks
    ]


def run_paper(arxiv_id: str, retriever: Retriever, full_texts: dict) -> list[dict]:
    qa_items = load_qa(arxiv_id)
    if not qa_items:
        return []

    results = []
    for qa_item in qa_items:
        question = qa_item["question"]
        logger.info(f"  Q: {question[:60]}")

        search_results = retriever.search(question, top_k=CONFIG["top_k"], use_rrf=True)
        hybrid_chunks = search_results["hybrid"]

        # offset
        expanded = [{**c, "text": expand_context(c, full_texts, CONFIG["context_offset"])} for c in hybrid_chunks]
        answer = generate_answer(question, expanded)

        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": qa_item.get("ground_truth", ""),
            "relevant_chunk_ids": qa_item.get("relevant_chunk_ids", []),
            "retrieved_chunks": format_chunks(hybrid_chunks),
            "bm25_chunks": format_chunks(search_results["bm25"]),
            "embedding_chunks": format_chunks(search_results["embedding"]),
        })

    return results


def run_all_papers() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

   
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # pronadji sve papere koji imaju i chunkove i QA - prvih 5
    arxiv_ids = sorted([
    f.replace("_chunks.json", "")
    for f in os.listdir(PROCESSED_DIR)
    if f.endswith("_chunks.json")
    ])[:5]  

    logger.info(f"Found {len(arxiv_ids)} papers")

    for arxiv_id in arxiv_ids:
        output_path = os.path.join(RESULTS_DIR, f"{arxiv_id}_results.json")
        if os.path.exists(output_path):
            logger.info(f"Skipping {arxiv_id} — already done")
            continue

        logger.info(f"\nProcessing {arxiv_id}...")
        chunks = load_chunks(arxiv_id)
        if not chunks:
            continue

        full_texts = build_full_text(chunks)

        # build retriever za ovaj paper
        os.environ["TOP_K"] = str(CONFIG["top_k"])
        os.environ["RRF_K"] = str(CONFIG["rrf_k"])
        retriever = Retriever()
        retriever.build_index(chunks)

        results = run_paper(arxiv_id, retriever, full_texts)
        if results:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(results)} results → {output_path}")

    logger.info("\nAll papers done!")


if __name__ == "__main__":
    run_all_papers()