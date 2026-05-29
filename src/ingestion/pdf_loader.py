import fitz

def _get_column(block, page_width: float) -> int:
    x0 = block[0]
    return 0 if x0 < page_width / 2 else 1

def load_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        blocks = sorted(blocks, key=lambda b: (_get_column(b, page_width), b[1]))
        page_text = " ".join(
            block[4].strip()
            for block in blocks
            if block[6] == 0
        )
        pages.append({
            "page_num": page.number,
            "text": page_text,
        })
    doc.close()
    return pages