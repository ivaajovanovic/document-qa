import sys
sys.path.append(".")

import os
import json
import uuid
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

PROCESSED_DIR = "./data/processed_256"

def load_chunks(processed_dir: str) -> list[dict]:
    chunks = []
    for fname in os.listdir(processed_dir):
        if fname.endswith(".json"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    chunks.extend(data)
    return chunks


def generate_questions_from_chunks(chunks: list[dict], n_questions: int = 20) -> list[dict]:
    import random
    random.seed(42)

    papers = {}
    for c in chunks:
        pid = c.get("metadata", {}).get("arxiv_id") or c.get("arxiv_id", "unknown")
        if pid not in papers:
            papers[pid] = []
        papers[pid].append(c)

    selected_chunks = []
    per_paper = max(1, n_questions // len(papers))
    for pid, paper_chunks in papers.items():
        mid = len(paper_chunks) // 4
        sample = paper_chunks[mid: mid + per_paper * 3]
        selected_chunks.extend(random.sample(sample, min(per_paper, len(sample))))

    selected_chunks = selected_chunks[:n_questions]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.environ["GROQ_API_KEY"]
    )

    results = []
    for i, chunk in enumerate(selected_chunks):
        text = chunk.get("text", "")
        if len(text.strip()) < 200:
            continue

        prompt = f"""You are creating evaluation data for a RAG system over academic NLP papers.

Given this passage from an academic paper, generate:
1. A specific factual question that can ONLY be answered using this passage
2. A concise ground truth answer based ONLY on this passage

Passage:
{text[:1000]}

Rules:
- The question must be answerable from the passage alone
- Prefer questions about methods, results, architectures, or comparisons
- Do NOT ask about page numbers or formatting
- Answer should be 1-3 sentences maximum

Respond ONLY with valid JSON, no markdown:
{{"question": "...", "ground_truth": "..."}}"""

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            parsed = json.loads(content.strip())

            results.append({
                "question": parsed["question"],
                "ground_truth": parsed["ground_truth"],
                "source_chunk": text[:500],
                "arxiv_id": chunk.get("metadata", {}).get("arxiv_id", chunk.get("arxiv_id", "")),
                "title": chunk.get("metadata", {}).get("title", chunk.get("title", "")),
            })
            print(f"  [{i+1}/{len(selected_chunks)}] ✓ {parsed['question'][:80]}")

        except Exception as e:
            print(f"  [{i+1}/{len(selected_chunks)}] ✗ Failed: {e}")
            continue

    return results


def run_rag_pipeline(questions: list[dict]) -> list[dict]:
    from src.langgraph_rag.graph import rag_graph
    from src.langgraph_rag.state import RAGState

    eval_results = []

    for i, item in enumerate(questions):
        question = item["question"]
        print(f"  [{i+1}/{len(questions)}] Running RAG: {question[:70]}")

        thread_id = str(uuid.uuid4())
        graph_config = {"configurable": {"thread_id": thread_id}}

        state: RAGState = {
            "messages": [HumanMessage(content=question)],
            "config_id": "config_256_k10_rrf60",
            "retrieved_chunks": None,
            "answer": None,
            "error": None,
        }

        try:
            result_state = rag_graph.invoke(state, config=graph_config)

            last_ai = next(
                (m for m in reversed(result_state["messages"]) if isinstance(m, AIMessage)),
                None
            )
            answer = last_ai.content if last_ai else ""

            contexts = []
            if result_state.get("retrieved_chunks"):
                hybrid = result_state["retrieved_chunks"].get("hybrid", [])
                contexts = [c["text"] for c in hybrid[:5]]

            eval_results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            })

        except Exception as e:
            print(f"    ✗ RAG failed: {e}")
            eval_results.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": item["ground_truth"],
            })

    return eval_results


def run_ragas_evaluation(eval_results: list[dict]) -> pd.DataFrame:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from langchain_ollama import OllamaEmbeddings
    from datasets import Dataset

    valid = [r for r in eval_results if r["answer"] and r["contexts"]]
    print(f"\nEvaluating {len(valid)}/{len(eval_results)} valid results...")

    dataset = Dataset.from_list(valid)

    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"]
        )
    )

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    run_config = RunConfig(max_retries=3, timeout=120)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    return result


if __name__ == "__main__":
    os.makedirs("./data/rag_eval", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    rag_results_path = "./data/rag_eval/rag_results.json"
    if os.path.exists(rag_results_path):
        print("Loading existing RAG results...")
        with open(rag_results_path, encoding="utf-8") as f:
            eval_results = json.load(f)
    else:
        print("Step 1: Loading chunks...")
        chunks = load_chunks(PROCESSED_DIR)
        print(f"  Loaded {len(chunks)} chunks")

        print("\nStep 2: Generating test questions...")
        questions = generate_questions_from_chunks(chunks, n_questions=20)
        with open("./data/rag_eval/generated_questions.json", "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        print("\nStep 3: Running RAG pipeline...")
        eval_results = run_rag_pipeline(questions)
        with open(rag_results_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print("\nStep 4: Running RAGAS evaluation...")
    result = run_ragas_evaluation(eval_results)
    print(result)
    result.to_pandas().to_csv("./results/ragas_evaluation.csv", index=False, encoding="utf-8")
    print("\nSaved to results/ragas_evaluation.csv")