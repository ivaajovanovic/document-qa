import sys
sys.path.append(".")

import os
import json
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.metadata import load_metadata
from src.ingestion.chunker import chunk_text


def ingest_pdf(pdf_path: str) -> list[dict]:
    """
    Ingests a single PDF and returns chunks with metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of chunks with text and metadata.
    """
    metadata = load_metadata(pdf_path)
    pages = load_pdf(pdf_path)  # text → pages
    chunks = chunk_text(pages, metadata)
    return chunks


def run_pipeline(
    data_dir: str = "./data/raw/arxiv_papers",
    output_dir: str = "./data/processed"
):
    """
    Runs ingestion pipeline for all PDFs and saves chunks as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files\n")

    total_chunks = 0
    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        arxiv_id = filename.replace(".pdf", "")
        print(f"Ingesting: {filename}...")

        chunks = ingest_pdf(pdf_path)

        output_path = os.path.join(output_dir, f"{arxiv_id}_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        total_chunks += len(chunks)
        print(f"  Saved {len(chunks)} chunks → {arxiv_id}_chunks.json")

    print(f"\nDone! Total {total_chunks} chunks saved to {output_dir}/")


if __name__ == "__main__":
    run_pipeline()