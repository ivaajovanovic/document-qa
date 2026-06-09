import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text by removing noise.
    """
    text = re.sub(r'arXiv:\S+', '', text)                         # arXiv ID oznake
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)              # reference [1], [2, 3]
    text = re.sub(r'[^\w\s.,;:!?()\-\'\"]+', ' ', text)          # specijalni karakteri
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(pages: list[dict], metadata: dict) -> list[dict]:
    """
    Cleans and splits text from pages into chunks with metadata.

    Args:
        pages: Output from load_pdf() - list of dicts with page_num and text.
        metadata: Dict from load_metadata() with title, authors, year...

    Returns:
        List of dicts, one per chunk, with keys:
            - text: cleaned chunk text
            - metadata: paper metadata + chunk_index + page_num + char positions
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=13,
    )

    chunks = []
    chunk_index = 0
    char_position = 0

    for page in pages:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue

        page_chunks = splitter.split_text(cleaned)

        for chunk in page_chunks:
            if not chunk.strip():
                continue

            chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "page_num": page["page_num"],
                    "char_start": char_position,
                    "char_end": char_position + len(chunk),
                }
            })

            char_position += len(chunk)
            chunk_index += 1

    return chunks