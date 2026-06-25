import sys
sys.path.append(".")

import os
import json
import logging
import time
from src.extraction.client import GroqClient
from src.extraction.extractor import extract_paper_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chunks(chunks_path: str) -> str:
    """
    Loads chunks from JSON file and joins text into a single string.

    Args:
        chunks_path: Path to chunks JSON file.

    Returns:
        Full paper text as a single string.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return " ".join(chunk["text"] for chunk in chunks)


def run_extraction(
    processed_dir: str = "./data/processed",
    output_dir: str = "./data/extracted_2"
):
    """
    Run extraction across all processed papers and persist JSON outputs.

    Workflow:
    1) Read ``*_chunks.json`` files from ``processed_dir``.
    2) Merge chunk text into one document per paper.
    3) Extract structured fields using the LLM client.
    4) Save ``*_extraction.json`` files to ``output_dir``.

    Note:
        Existing output files are skipped, so interrupted runs can continue.
    """
    os.makedirs(output_dir, exist_ok=True)
    client = GroqClient()

    chunk_files = [f for f in os.listdir(processed_dir) if f.endswith("_chunks.json")]
    logger.info(f"Found {len(chunk_files)} papers to process")

    for filename in chunk_files:
        arxiv_id = filename.replace("_chunks.json", "")
        output_path = os.path.join(output_dir, f"{arxiv_id}_extraction.json")

        if os.path.exists(output_path):
            logger.info(f"Skipping {arxiv_id} — already extracted")
            continue

        logger.info(f"Extracting: {arxiv_id}...")
        chunks_path = os.path.join(processed_dir, filename)
        # Rebuild a single text blob used as extraction context.
        text = load_chunks(chunks_path)

        extraction = extract_paper_info(text, client)

        if extraction:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(extraction.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved → {arxiv_id}_extraction.json")
        else:
            logger.error(f"  Failed to extract: {arxiv_id}")

        # Conservative delay to reduce rate-limit issues on hosted APIs.
        time.sleep(65)

    logger.info("Done!")


if __name__ == "__main__":
    run_extraction()