import json
import os


def load_metadata(pdf_path: str) -> dict:
    """
    Loads metadata from a JSON file located next to the PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with metadata: arxiv_id, title, authors, year, categories.
    """
    json_path = pdf_path.replace(".pdf", ".json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)