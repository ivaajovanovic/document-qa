import sys
sys.path.append(".")

import os
import json
import logging
import unicodedata
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    arxiv_id: str
    title_match: bool
    year_match: bool
    author_match: float
    keyword_overlap: float
    hallucinations: list[str]


def normalize(text: str) -> str:
    """Normalize text by removing special characters and lowercasing."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def evidence_in_text(evidence: str, text: str, threshold: float = 0.7) -> bool:
    """Check if evidence appears in text using fuzzy word matching."""
    if not evidence or evidence == "not found":
        return True
    evidence_words = [w for w in normalize(evidence).split() if len(w) > 3]
    if not evidence_words:
        return True
    text_normalized = normalize(text)
    matches = sum(1 for w in evidence_words if w in text_normalized)
    return matches / len(evidence_words) >= threshold


def exact_match_title(extracted: str, expected: str) -> bool:
    return extracted.strip().lower() == expected.strip().lower()


def exact_match_year(extracted: str, expected: str) -> bool:
    return extracted.strip() == expected.strip()


def author_match(extracted: list[str], expected: list[str]) -> float:
    """Returns percentage of expected authors found in extracted."""
    if not expected:
        return 1.0
    extracted_normalized = [normalize(a) for a in extracted]
    matches = sum(
        1 for author in expected
        if any(
            normalize(author) in e or e in normalize(author)
            for e in extracted_normalized
        )
    )
    return matches / len(expected)


def keyword_overlap(extracted_keywords: list[str], extracted_topic: str, paper_text: str) -> float:
    """Returns percentage of extracted keywords/topic found in paper text."""
    if not extracted_keywords:
        return 0.0
    all_extracted = [k.lower() for k in extracted_keywords] + [extracted_topic.lower()]
    text_lower = paper_text.lower()
    matches = sum(1 for kw in all_extracted if kw in text_lower)
    return matches / len(all_extracted)


def detect_hallucinations(extracted: dict, expected: dict, paper_text: str) -> list[str]:
    """
    Detects hallucinations by comparing extracted data with known metadata
    and checking if extracted info appears in paper text.
    """
    hallucinations = []

    # year mismatch
    if extracted.get("year") != expected.get("year"):
        hallucinations.append(
            f"Year mismatch: extracted={extracted.get('year')}, expected={expected.get('year')}"
        )

    # year evidence not in text
    year_evidence = extracted.get("year_evidence", "")
    if year_evidence and year_evidence != "not found":
        if not evidence_in_text(year_evidence, paper_text):
            hallucinations.append(f"Year evidence not found in text: {year_evidence[:50]}...")

    # unknown authors — normalized comparison
    expected_authors_normalized = [normalize(a) for a in expected.get("authors", [])]
    for author in extracted.get("authors", []):
        author_norm = normalize(author)
        if not any(
            author_norm in ea or ea in author_norm
            for ea in expected_authors_normalized
        ):
            hallucinations.append(f"Unknown author: {author}")

    # authors evidence — fuzzy match
    authors_evidence = extracted.get("authors_evidence", "")
    if authors_evidence and not evidence_in_text(authors_evidence, paper_text):
        hallucinations.append(f"Authors evidence not found in text: {authors_evidence[:50]}...")

    # companies not found in text — normalized comparison
    text_normalized = normalize(paper_text)
    for company in extracted.get("companies", []):
        if normalize(company) not in text_normalized:
            hallucinations.append(f"Company not found in text: {company}")

    # methodology evidence — fuzzy match
    methodology_evidence = extracted.get("methodology_evidence", "")
    if methodology_evidence and not evidence_in_text(methodology_evidence, paper_text):
        hallucinations.append(f"Methodology evidence not found in text: {methodology_evidence[:50]}...")

    return hallucinations


def evaluate_extraction(
    extracted_dir: str = "./data/extracted_2",
    processed_dir: str = "./data/processed",
    metadata_dir: str = "./data/raw/arxiv_papers",
    output_path: str = "./results/evaluation2.json"
):
    os.makedirs("./results", exist_ok=True)
    results = []

    extraction_files = [f for f in os.listdir(extracted_dir) if f.endswith("_extraction.json")]
    logger.info(f"Evaluating {len(extraction_files)} extractions...")

    for filename in extraction_files:
        arxiv_id = filename.replace("_extraction.json", "")

        with open(os.path.join(extracted_dir, filename), "r", encoding="utf-8") as f:
            extracted = json.load(f)

        metadata_path = os.path.join(metadata_dir, f"{arxiv_id}.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"No metadata found for {arxiv_id}")
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        chunks_path = os.path.join(processed_dir, f"{arxiv_id}_chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        full_text = " ".join(chunk["text"] for chunk in chunks)

        result = EvaluationResult(
            arxiv_id=arxiv_id,
            title_match=exact_match_title(extracted.get("title", ""), expected.get("title", "")),
            year_match=exact_match_year(extracted.get("year", ""), expected.get("year", "")),
            author_match=author_match(extracted.get("authors", []), expected.get("authors", [])),
            keyword_overlap=keyword_overlap(
                extracted.get("keywords", []),
                extracted.get("primary_topic", ""),
                full_text
            ),
            hallucinations=detect_hallucinations(extracted, expected, full_text)
        )

        results.append(asdict(result))
        logger.info(
            f"{arxiv_id}: title={result.title_match}, year={result.year_match}, "
            f"authors={result.author_match:.2f}, keywords={result.keyword_overlap:.2f}, "
            f"hallucinations={len(result.hallucinations)}"
        )

    if results:
        summary = {
            "total": len(results),
            "title_match_rate": sum(r["title_match"] for r in results) / len(results),
            "year_match_rate": sum(r["year_match"] for r in results) / len(results),
            "avg_author_match": sum(r["author_match"] for r in results) / len(results),
            "avg_keyword_overlap": sum(r["keyword_overlap"] for r in results) / len(results),
            "total_hallucinations": sum(len(r["hallucinations"]) for r in results),
            "papers": results
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\nSummary:")
        logger.info(f"  Title match rate: {summary['title_match_rate']:.2%}")
        logger.info(f"  Year match rate: {summary['year_match_rate']:.2%}")
        logger.info(f"  Avg author match: {summary['avg_author_match']:.2%}")
        logger.info(f"  Avg keyword overlap: {summary['avg_keyword_overlap']:.2%}")
        logger.info(f"  Total hallucinations: {summary['total_hallucinations']}")
        logger.info(f"  Results saved to {output_path}")


if __name__ == "__main__":
    evaluate_extraction()