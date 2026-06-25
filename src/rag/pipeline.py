import sys
sys.path.append(".")

"""End-to-end prototype RAG pipeline script.

This file was used as a practical first pass: load chunks, build retriever,
ask a few questions, and save outputs for manual inspection.
"""

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

CONTEXT_OFFSET = int(os.getenv("CONTEXT_OFFSET", "150"))


def load_chunks(processed_dir: str = "./data/processed", arxiv_id: str = None) -> list[dict]:
    """
    Load chunk files for one paper or the whole processed folder.

    Args:
        processed_dir: Path to processed chunks directory.
        arxiv_id: If provided, load only chunks for this paper.

    Returns:
        List of chunks.
    """
    all_chunks = []
    for filename in os.listdir(processed_dir):
        if not filename.endswith("_chunks.json"):
            continue
        if arxiv_id and arxiv_id not in filename:
            continue
        with open(os.path.join(processed_dir, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    logger.info(f"Loaded {len(all_chunks)} chunks")
    return all_chunks


def build_full_text(chunks: list[dict]) -> dict[str, str]:
    """
    Reconstruct approximate full text per paper from chunk order.

    Args:
        chunks: List of all chunks.

    Returns:
        Dict mapping arxiv_id to full text.
    """
    full_texts = {}
    for chunk in chunks:
        arxiv_id = chunk["metadata"].get("arxiv_id")
        if arxiv_id not in full_texts:
            full_texts[arxiv_id] = ""
        full_texts[arxiv_id] += chunk["text"]
    return full_texts


def expand_context(chunk: dict, full_texts: dict, offset: int = CONTEXT_OFFSET) -> str:
    """
    Expand a chunk with neighboring characters to add local context.

    Args:
        chunk: Chunk dict sa metadata.
        full_texts: Dict mapping arxiv_id to full text.
        offset: Broj karaktera pre i posle chunka.

    Returns:
        Prošireni tekst.
    """
    arxiv_id = chunk["metadata"].get("arxiv_id")
    char_start = chunk["metadata"].get("char_start", 0)
    char_end = chunk["metadata"].get("char_end", len(chunk["text"]))

    full_text = full_texts.get(arxiv_id, chunk["text"])

    expanded_start = max(0, char_start - offset)
    expanded_end = min(len(full_text), char_end + offset)

    return full_text[expanded_start:expanded_end]


class RAGPipeline:
    """
    Simple wrapper that exposes `build()` and `ask()` for prototype runs.
    """

    def __init__(self, processed_dir: str = "./data/processed"):
        self.retriever = Retriever()
        self.processed_dir = processed_dir
        self._built = False
        self.chunks = []
        self.full_texts = {}

    def build(self, arxiv_id: str = None) -> None:
        """
        Load chunks and initialize retriever index for querying.

        Args:
            arxiv_id: If provided, build index only for this paper.
        """
        logger.info("Building RAG pipeline...")
        self.chunks = load_chunks(self.processed_dir, arxiv_id=arxiv_id)
        self.full_texts = build_full_text(self.chunks)
        self.retriever.build_index(self.chunks)
        self._built = True
        logger.info("RAG pipeline ready")

    def ask(self, question: str, top_k: int = 5, use_rrf: bool = True, use_offset: bool = True) -> dict:
        """
        Answer one question via retrieve-then-generate flow.

        Args:
            question: User question.
            top_k: Number of chunks to retrieve.
            use_rrf: If True uses RRF fusion, if False returns union of both retrievers.
            use_offset: If True expands chunk context with offset.

        Returns:
            Dict with answer, hybrid chunks, bm25 chunks and embedding chunks.
        """
        if not self._built:
            logger.error("Pipeline not built — call build() first")
            return {}

        search_results = self.retriever.search(question, top_k=top_k, use_rrf=use_rrf)
        hybrid_chunks = search_results["hybrid"]

        # prosiri kontekst sa offsetom
        if use_offset:
            for chunk in hybrid_chunks:
                chunk["expanded_text"] = expand_context(chunk, self.full_texts)
            context_chunks = [{**c, "text": c["expanded_text"]} for c in hybrid_chunks]
        else:
            context_chunks = hybrid_chunks

        answer = generate_answer(question, context_chunks)

        def format_chunks(chunks):
            return [
                {
                    "text": c["text"],
                    "score": c.get("score"),
                    "title": c["metadata"].get("title"),
                    "arxiv_id": c["metadata"].get("arxiv_id"),
                    "page_num": c["metadata"].get("page_num")
                }
                for c in chunks
            ]

        return {
            "question": question,
            "answer": answer,
            "mode": search_results["mode"],
            "use_offset": use_offset,
            "retrieved_chunks": format_chunks(hybrid_chunks),
            "bm25_chunks": format_chunks(search_results["bm25"]),
            "embedding_chunks": format_chunks(search_results["embedding"])
        }


def save_results(results: list[dict], output_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    ARXIV_ID = "1706.03762"

    for chunk_size in ["256", "128"]:
        processed_dir = f"./data/processed_{chunk_size}"
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with chunk_size={chunk_size}")

        pipeline = RAGPipeline(processed_dir=processed_dir)
        pipeline.build(arxiv_id=ARXIV_ID)

        questions = [
            "What is the Transformer architecture?",
            "How does multi-head attention work?",
            "What is positional encoding and why is it needed?",
            "What tasks was the Transformer evaluated on?",
            "How does scaled dot-product attention work?",
        ]

        for use_rrf in [True, False]:
            mode = "rrf" if use_rrf else "union"
            results = []

            for question in questions:
                logger.info(f"Asking: {question}")
                result = pipeline.ask(question, top_k=5, use_rrf=use_rrf, use_offset=True)
                results.append(result)
                print(f"\nQ: {result['question']}")
                print(f"A: {result['answer']}")
                print(f"Hybrid:    {[(c['page_num'], c['score']) for c in result['retrieved_chunks']]}")
                print(f"BM25:      {[(c['page_num'], c['score']) for c in result['bm25_chunks']]}")
                print(f"Embedding: {[(c['page_num'], c['score']) for c in result['embedding_chunks']]}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_results(results, f"./results/rag_{ARXIV_ID}_{chunk_size}_{mode}_{timestamp}.json")