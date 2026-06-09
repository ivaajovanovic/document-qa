import sys
sys.path.append(".")

import os
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "./results"
QA_PATH = "./data/rag_eval/1706.03762_qa.json"


class GroqEvaluator(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(5)
                response = await self.model.ainvoke(prompt)
                return response.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    def get_model_name(self) -> str:
        return "llama-3.3-70b-versatile"


def load_qa_dataset(qa_path: str) -> dict:
    """Ucitava QA dataset i vraca dict question -> ground_truth."""
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["question"]: item["ground_truth"] for item in data}


def compute_recall_at_k(retrieved_chunks: list[dict], ground_truth: str, k: int) -> float:
    """
    Racuna Recall@K — da li ground truth koncepti postoje u top K retrieved chunkovima.
    Koristi keyword overlap kao proxy.
    """
    if not retrieved_chunks:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    combined_text = " ".join(c["text"].lower() for c in top_k_chunks)

    gt_words = [w for w in ground_truth.lower().split() if len(w) > 4]
    if not gt_words:
        return 0.0

    matches = sum(1 for w in gt_words if w in combined_text)
    return round(matches / len(gt_words), 3)


def evaluate_experiment(
    exp_dir: str,
    qa_data: dict,
    evaluator: GroqEvaluator
) -> dict:
    """
    Evaluira jedan eksperiment sa svim retrieval modovima.

    Args:
        exp_dir: Path to experiment directory.
        qa_data: Dict question -> ground_truth.
        evaluator: GroqEvaluator instance.

    Returns:
        Dict sa metrikama za svaki mod.
    """
    with open(os.path.join(exp_dir, "config.json"), "r") as f:
        config = json.load(f)

    metrics_all = {"config": config, "modes": {}}

    for mode in ["rrf", "bm25", "embedding"]:
        results_path = os.path.join(exp_dir, f"results_{mode}.json")
        if not os.path.exists(results_path):
            logger.warning(f"Missing {results_path}")
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        logger.info(f"Evaluating {os.path.basename(exp_dir)} — {mode}")

        test_cases = []
        recall_scores = []

        for result in results:
            question = result["question"]
            ground_truth = qa_data.get(question, "")
            if not ground_truth:
                continue

            retrieved_chunks = result["retrieved_chunks"]

            # Recall@K
            recall = compute_recall_at_k(
                retrieved_chunks,
                ground_truth,
                k=config.get("top_k", 5)
            )
            recall_scores.append(recall)

            # DeepEval test case
            test_cases.append(LLMTestCase(
                input=question,
                actual_output=result["answer"],
                expected_output=ground_truth,
                retrieval_context=[c["text"] for c in retrieved_chunks]
            ))

        # DeepEval metrike
        deepeval_metrics = [
            FaithfulnessMetric(model=evaluator, threshold=0.5, async_mode=False),
            AnswerRelevancyMetric(model=evaluator, threshold=0.5, async_mode=False),
            ContextualPrecisionMetric(model=evaluator, threshold=0.5, async_mode=False),
            ContextualRecallMetric(model=evaluator, threshold=0.5, async_mode=False),
        ]

        eval_results = evaluate(test_cases, deepeval_metrics)

        scores = {}
        for test_result in eval_results.test_results:
            for metric_data in test_result.metrics_data:
                name = metric_data.name
                if name not in scores:
                    scores[name] = []
                scores[name].append(metric_data.score)

        scores = {name: round(sum(vals) / len(vals), 3) for name, vals in scores.items()}
        scores["RecallAtK"] = round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else 0.0

        metrics_all["modes"][mode] = scores
        logger.info(f"  {mode}: {scores}")

    # sacuvaj metrike
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {metrics_path}")

    return metrics_all


def run_evaluation(results_dir: str = RESULTS_DIR, qa_path: str = QA_PATH) -> None:
    """Evaluira sve eksperimente i pravi summary."""
    qa_data = load_qa_dataset(qa_path)
    evaluator = GroqEvaluator()

    exp_dirs = sorted([
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("exp_")
    ])

    logger.info(f"Found {len(exp_dirs)} experiments to evaluate")

    all_metrics = []
    for exp_dir in exp_dirs:
        try:
            metrics = evaluate_experiment(exp_dir, qa_data, evaluator)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Failed to evaluate {exp_dir}: {e}")
            continue

    # sacuvaj summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run_evaluation()