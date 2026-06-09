import sys
sys.path.append(".")

import os
import json
import logging
from langchain_ollama import ChatOllama

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "./results"
QA_DIR = "./data/rag_eval"
ALL_PAPERS_RESULTS_DIR = "./results/exp_all_256_k10_rrf60"

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="llama3.2")
    return _llm


def load_qa_dataset(qa_dir: str) -> dict:
    """Load all QA files and return dict: question -> {ground_truth, relevant_chunk_ids}."""
    qa_data = {}
    for filename in os.listdir(qa_dir):
        if not filename.endswith("_qa.json"):
            continue
        with open(os.path.join(qa_dir, filename), "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = [items]
        for item in items:
            qa_data[item["question"]] = {
                "ground_truth": item.get("ground_truth", ""),
                "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
            }
    logger.info(f"Loaded {len(qa_data)} QA pairs")
    return qa_data


def tokenize(text: str) -> set:
    """Tokenize text — return set of words longer than 3 characters."""
    return set(w.lower() for w in text.split() if len(w) > 3)


def recall_at_k(retrieved_chunks: list[dict], relevant_chunk_ids: list[int], k: int) -> float:
    """
    Exact Recall@K = |relevant ∩ retrieved[:K]| / |relevant|

    Args:
        retrieved_chunks: List of retrieved chunks with metadata.
        relevant_chunk_ids: List of chunk_index values of relevant chunks.
        k: Number of top chunks to consider.

    Returns:
        Float 0-1.
    """
    if not retrieved_chunks or not relevant_chunk_ids:
        return 0.0

    retrieved_ids = set(
        c["metadata"]["chunk_index"]
        for c in retrieved_chunks[:k]
        if "chunk_index" in c.get("metadata", {})
    )
    relevant_ids = set(relevant_chunk_ids)
    retrieved_relevant = retrieved_ids & relevant_ids

    return round(len(retrieved_relevant) / len(relevant_ids), 3)


def context_precision(retrieved_chunks: list[dict], relevant_chunk_ids: list[int]) -> float:
    """
    Exact Context Precision = relevant retrieved chunks / total retrieved chunks.
    Uses relevant_chunk_ids instead of keyword overlap.

    Args:
        retrieved_chunks: List of retrieved chunks with metadata.
        relevant_chunk_ids: List of chunk_index values of relevant chunks.

    Returns:
        Float 0-1.
    """
    if not retrieved_chunks or not relevant_chunk_ids:
        return 0.0

    relevant_ids = set(relevant_chunk_ids)
    retrieved_ids = [
        c["metadata"].get("chunk_index")
        for c in retrieved_chunks
        if "chunk_index" in c.get("metadata", {})
    ]

    relevant_retrieved = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return round(relevant_retrieved / len(retrieved_chunks), 3)


def extract_claims(answer: str) -> list[str]:
    """Extract factual claims from answer using LLM."""
    prompt = f"""Extract all factual claims from this answer as a JSON array of strings.
Each claim should be a single, verifiable statement.

Answer: {answer}

Respond with ONLY a JSON array, no other text:
["claim1", "claim2", ...]"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        text = response.content.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        claims = json.loads(text[start:end])
        return [c for c in claims if isinstance(c, str) and len(c) > 5]
    except Exception as e:
        logger.warning(f"Failed to extract claims: {e}")
        return []


def verify_claim(claim: str, context: str) -> bool:
    """Check whether a claim is supported by the context using LLM."""
    prompt = f"""Is the following claim supported by the context? Answer with only "yes" or "no".

Context: {context}

Claim: {claim}"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return "yes" in response.content.lower()
    except Exception as e:
        logger.warning(f"Failed to verify claim: {e}")
        return False


def answer_faithfulness(answer: str, retrieved_chunks: list[dict]) -> float:
    """
    LLM-as-judge faithfulness — decomposes answer into claims and verifies each one.
    Score = supported claims / total claims.

    Falls back to keyword overlap if claim extraction fails.

    Args:
        answer: Generated answer.
        retrieved_chunks: List of retrieved chunks used as context.

    Returns:
        Float 0-1.
    """
    if not answer or not retrieved_chunks:
        return 0.0

    context = "\n\n".join(c["text"] for c in retrieved_chunks)

    # step 1 — extract claims from answer
    claims = extract_claims(answer)
    if not claims:
        logger.warning("No claims extracted, falling back to keyword overlap")
        context_tokens = tokenize(context)
        answer_tokens = tokenize(answer)
        if not answer_tokens:
            return 0.0
        matches = sum(1 for w in answer_tokens if w in context_tokens)
        return round(matches / len(answer_tokens), 3)

    logger.info(f"  Verifying {len(claims)} claims...")

    # step 2 — verify each claim against context
    supported = sum(1 for claim in claims if verify_claim(claim, context))

    score = round(supported / len(claims), 3)
    logger.info(f"  Faithfulness: {supported}/{len(claims)} claims supported = {score}")
    return score


def evaluate_results(results: list[dict], qa_data: dict, top_k: int) -> dict:
    """
    Evaluate a list of results with all metrics.

    Args:
        results: List of results from results_*.json.
        qa_data: Dict question -> {ground_truth, relevant_chunk_ids}.
        top_k: K value for Recall@K.

    Returns:
        Dict with per-question metrics and averages.
    """
    per_question = []
    recall_scores = []
    precision_scores = []
    faithfulness_scores = []

    for result in results:
        question = result["question"]
        answer = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        qa_item = qa_data.get(question)

        if not qa_item:
            logger.warning(f"No QA data for: {question}")
            continue

        ground_truth = qa_item["ground_truth"]
        relevant_chunk_ids = qa_item["relevant_chunk_ids"]

        r_at_k = recall_at_k(retrieved_chunks, relevant_chunk_ids, k=top_k)
        c_prec = context_precision(retrieved_chunks, relevant_chunk_ids)
        a_faith = answer_faithfulness(answer, retrieved_chunks)

        recall_scores.append(r_at_k)
        precision_scores.append(c_prec)
        faithfulness_scores.append(a_faith)

        logger.info(f"  R@K={r_at_k} Prec={c_prec} Faith={a_faith} — {question[:50]}")

        per_question.append({
            "question": question,
            "recall_at_k": r_at_k,
            "context_precision": c_prec,
            "answer_faithfulness": a_faith,
        })

    return {
        "per_question": per_question,
        "averages": {
            "recall_at_k": round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0,
            "context_precision": round(sum(precision_scores) / len(precision_scores), 3) if precision_scores else 0.0,
            "answer_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else 0.0,
        }
    }


def evaluate_experiment(exp_dir: str, qa_data: dict) -> dict:
    """Evaluate one experiment across all retrieval modes (rrf, bm25, embedding)."""
    with open(os.path.join(exp_dir, "config.json"), "r") as f:
        config = json.load(f)

    top_k = config.get("top_k", 5)
    metrics_all = {"config": config, "modes": {}}

    for mode in ["rrf", "bm25", "embedding"]:
        results_path = os.path.join(exp_dir, f"results_{mode}.json")
        if not os.path.exists(results_path):
            logger.warning(f"Missing {results_path}")
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        metrics = evaluate_results(results, qa_data, top_k)
        metrics_all["modes"][mode] = metrics

        logger.info(f"{os.path.basename(exp_dir)} — {mode}: {metrics['averages']}")

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

    return metrics_all


def run_evaluation(results_dir: str = RESULTS_DIR, qa_dir: str = QA_DIR) -> None:
    """Evaluate experiments (exp_001 to exp_006) and save summary."""
    qa_data = load_qa_dataset(qa_dir)

    exp_dirs = sorted([
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("exp_")
        and not d == "exp_all_256_k10_rrf60"
    ])

    logger.info(f"Found {len(exp_dirs)} experiments")

    all_metrics = []
    for exp_dir in exp_dirs:
        try:
            metrics = evaluate_experiment(exp_dir, qa_data)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Failed {exp_dir}: {e}")
            continue

        # save summary after each experiment so progress is not lost
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary updated")

    logger.info(f"Done! Summary saved to {results_dir}/summary.json")


def run_evaluation_all_papers(
    results_dir: str = ALL_PAPERS_RESULTS_DIR,
    qa_dir: str = QA_DIR
) -> None:
    """Evaluate results across all papers from run_all_papers.py."""
    qa_data = load_qa_dataset(qa_dir)

    result_files = sorted([
        f for f in os.listdir(results_dir)
        if f.endswith("_results.json")
    ])

    logger.info(f"Found {len(result_files)} paper results")

    all_metrics = []
    for filename in result_files:
        arxiv_id = filename.replace("_results.json", "")
        with open(os.path.join(results_dir, filename), "r", encoding="utf-8") as f:
            results = json.load(f)

        metrics = evaluate_results(results, qa_data, top_k=10)
        metrics["arxiv_id"] = arxiv_id
        all_metrics.append(metrics)
        logger.info(f"{arxiv_id}: {metrics['averages']}")

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["experiments", "all_papers"], default="experiments")
    args = parser.parse_args()

    if args.mode == "all_papers":
        run_evaluation_all_papers()
    else:
        run_evaluation()