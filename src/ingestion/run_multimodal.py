import os
import json
from dotenv import load_dotenv
from pdf_loader_multimodal import load_pdf, extract_images_to_disk
from chunker_multimodal import chunk_text_multimodal
from metadata import load_metadata

load_dotenv()

RAW_DIR = "./data/raw/arxiv_papers"
FIGURES_DIR = "./data/figures"
OUTPUT_DIR = "./data/processed_multimodal"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_paper(filename: str) -> None:
    """Process one PDF into multimodal chunks and save JSON output."""
    arxiv_id = filename.replace(".pdf", "")
    pdf_path = os.path.join(RAW_DIR, filename)
    figures_dir = os.path.join(FIGURES_DIR, arxiv_id)
    output_path = os.path.join(OUTPUT_DIR, f"{arxiv_id}_chunks.json")

    # Skip already processed papers to support resumable runs.
    if os.path.exists(output_path):
        print(f"Skipping {arxiv_id} (already processed)")
        return

    print(f"Processing {arxiv_id}...")

    try:
        metadata = load_metadata(pdf_path)
        pages = load_pdf(pdf_path)
        pages = extract_images_to_disk(pdf_path, pages, figures_dir)

        total_images = sum(len(p["images"]) for p in pages)
        print(f"  Found {len(pages)} pages, {total_images} images")
        for p in pages:
            if p["images"]:
                print(f"  Page {p['page_num']}: {len(p['images'])} images")
                for img in p["images"]:
                    cap = img["caption"][:80] if img["caption"] else "NO CAPTION"
                    print(f"    [{img['width']}x{img['height']}] caption: {cap}")

        chunks = chunk_text_multimodal(pages, metadata, generate_descriptions=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        text_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "text"]
        fig_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "figure"]

        print(f"  Text chunks: {len(text_chunks)}, Figure chunks: {len(fig_chunks)}")
        print(f"  Saved to: {output_path}")

        if fig_chunks:
            print("  Sample figures:")
            for fc in fig_chunks[:2]:
                print(f"    [{fc['metadata']['figure_id']}] {fc['metadata']['llm_description'][:80]}")

        print()

    except Exception as e:
        print(f"  ERROR {arxiv_id}: {e}")
        import traceback
        traceback.print_exc()
        print()


def run_multimodal_pipeline() -> None:
    """Iterate over all PDFs and process each paper through the multimodal pipeline."""
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".pdf"):
            process_paper(filename)


if __name__ == "__main__":
    run_multimodal_pipeline()