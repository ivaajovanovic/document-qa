import sys
sys.path.append(".")

import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from src.rag.retriever import Retriever
from src.rag.generator import generate_answer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chunks(processed_dir: str, arxiv_id: str = None) -> list[dict]:
    """Load chunks from processed directory."""
    all_chunks = []
    for filename in os.listdir(processed_dir):
        if not filename.endswith("_chunks.json"):
            continue
        if arxiv_id and arxiv_id not in filename:
            continue
        with open(os.path.join(processed_dir, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    logger.info(f"Loaded {len(all_chunks)} chunks from {processed_dir}")
    return all_chunks


def build_full_text(chunks: list[dict]) -> dict[str, str]:
    """Rekonstruiše ceo tekst papera iz chunkova."""
    full_texts = {}
    for chunk in chunks:
        arxiv_id = chunk["metadata"].get("arxiv_id")
        if arxiv_id not in full_texts:
            full_texts[arxiv_id] = ""
        full_texts[arxiv_id] += chunk["text"]
    return full_texts


def expand_context(chunk: dict, full_texts: dict, offset: int) -> str:
    """Prosiruje tekst chunka sa offsetom pre i posle."""
    arxiv_id = chunk["metadata"].get("arxiv_id")
    char_start = chunk["metadata"].get("char_start", 0)
    char_end = chunk["metadata"].get("char_end", len(chunk["text"]))
    full_text = full_texts.get(arxiv_id, chunk["text"])
    expanded_start = max(0, char_start - offset)
    expanded_end = min(len(full_text), char_end + offset)
    return full_text[expanded_start:expanded_end]


def format_chunks(chunks: list[dict]) -> list[dict]:
    """Formatira chunkove za output."""
    return [
        {
            "text": c["text"],
            "score": c.get("score"),
            "title": c["metadata"].get("title"),
            "arxiv_id": c["metadata"].get("arxiv_id"),
            "page_num": c["metadata"].get("page_num"),
            "char_start": c["metadata"].get("char_start"),
            "char_end": c["metadata"].get("char_end"),
        }
        for c in chunks
    ]


def run_experiment(config: dict, questions: list[str], arxiv_id: str) -> None:
    """
    Runs a single experiment with given config.

    Args:
        config: Experiment config with id, chunk_size, top_k, rrf_k, context_offset.
        questions: List of questions to ask.
        arxiv_id: Paper to evaluate on.
    """
    exp_id = config["id"]
    chunk_size = config["chunk_size"]
    top_k = config["top_k"]
    rrf_k = config["rrf_k"]
    offset = config["context_offset"]

    processed_dir = f"./data/processed_{chunk_size}"
    output_dir = f"./results/{exp_id}_chunk{chunk_size}_k{top_k}_rrf{rrf_k}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Running {exp_id}: chunk={chunk_size}, k={top_k}, rrf_k={rrf_k}, offset={offset}")

    # sacuvaj config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({**config, "arxiv_id": arxiv_id, "timestamp": datetime.now().isoformat()}, f, indent=2)

    # ucitaj chunkove
    chunks = load_chunks(processed_dir, arxiv_id=arxiv_id)
    full_texts = build_full_text(chunks)

    # build retriever
    os.environ["TOP_K"] = str(top_k)
    os.environ["RRF_K"] = str(rrf_k)
    retriever = Retriever()
    retriever.build_index(chunks)

    results_rrf = []
    results_bm25 = []
    results_embedding = []

    for question in questions:
        logger.info(f"Q: {question}")

        search_results = retriever.search(question, top_k=top_k, use_rrf=True)
        bm25_chunks = search_results["bm25"]
        embedding_chunks = search_results["embedding"]
        hybrid_chunks = search_results["hybrid"]

        # prosiri kontekst sa offsetom
        def with_offset(chunks):
            expanded = []
            for c in chunks:
                expanded.append({**c, "text": expand_context(c, full_texts, offset)})
            return expanded

        bm25_expanded = with_offset(bm25_chunks)
        embedding_expanded = with_offset(embedding_chunks)
        hybrid_expanded = with_offset(hybrid_chunks)

        # generiši odgovore
        answer_rrf = generate_answer(question, hybrid_expanded)
        answer_bm25 = generate_answer(question, bm25_expanded)
        answer_embedding = generate_answer(question, embedding_expanded)

        results_rrf.append({
            "question": question,
            "answer": answer_rrf,
            "retrieved_chunks": format_chunks(hybrid_chunks),
        })
        results_bm25.append({
            "question": question,
            "answer": answer_bm25,
            "retrieved_chunks": format_chunks(bm25_chunks),
        })
        results_embedding.append({
            "question": question,
            "answer": answer_embedding,
            "retrieved_chunks": format_chunks(embedding_chunks),
        })

    # sacuvaj rezultate
    for name, results in [("rrf", results_rrf), ("bm25", results_bm25), ("embedding", results_embedding)]:
        path = os.path.join(output_dir, f"results_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {path}")

    logger.info(f"Experiment {exp_id} done → {output_dir}")