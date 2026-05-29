import requests
import json
import xml.etree.ElementTree as ET
import os
import time

# Lista svih 20 paper-ova (ID iz arxiv URL-a)
PAPER_IDS = [
    # Transformeri
    "1706.03762",  # Attention Is All You Need
    "2311.17633",  # Introduction to Transformers: an NLP Perspective
    "2503.20227",  # Advancements in NLP: Transformer-Based Architectures

    # Text Classification
    "1904.08067",  # Text Classification Algorithms: A Survey
    "2004.03705",  # Deep Learning Based Text Classification
    "2204.03954",  # Are We Really Making Much Progress in Text Classification?

    # RAG
    "2312.10997",  # RAG for LLMs: A Survey
    "2404.10981",  # Survey on Retrieval-Augmented Text Generation
    "2402.19473",  # RAG for AI-Generated Content

    # Embeddings
    "1908.10084",  # Sentence-BERT
    "2408.08073",  # Extracting Sentence Embeddings from Pretrained Transformers
    "2204.00820",  # Efficient comparison of sentence embeddings

    # BERT i Language Models
    "1810.04805",  # BERT
    "1907.11692",  # RoBERTa
    "2303.18223",  # A Survey on Large Language Models

    # Sentiment Analysis
    "2305.14842",  # Exploring Sentiment Analysis Techniques in NLP: A Comprehensive Review
    "2409.09989",  # Comprehensive Study on Sentiment Analysis: From Rule-based to modern LLM based system
    "2203.01054",  # Aspect-Based Sentiment Analysis

    # Contrastive Learning
    "2104.08821",  # SimCSE
    "2011.00362",  # Survey on Contrastive Self-supervised Learning
]

NAMESPACE = "{http://www.w3.org/2005/Atom}"

def fetch_metadata(paper_id: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
    response = requests.get(url)
    root = ET.fromstring(response.content)

    entry = root.find(f"{NAMESPACE}entry")
    if entry is None:
        print(f"  [!] Nije pronađen entry za {paper_id}")
        return {}

    title = entry.find(f"{NAMESPACE}title").text.strip().replace("\n", " ")
    year = entry.find(f"{NAMESPACE}published").text[:4]
    authors = [
        a.find(f"{NAMESPACE}name").text
        for a in entry.findall(f"{NAMESPACE}author")
    ]
    categories = [
        tag.attrib.get("term")
        for tag in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
    ] + [
        tag.attrib.get("term")
        for tag in entry.findall("{http://www.w3.org/2005/Atom}category")
    ]
    categories = list(set(filter(None, categories)))

    return {
        "arxiv_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "categories": categories,
    }


def main():
    output_dir = "data/raw/arxiv_papers"
    os.makedirs(output_dir, exist_ok=True)

    for paper_id in PAPER_IDS:
        print(f"Fetchujem: {paper_id}...")
        metadata = fetch_metadata(paper_id)

        if metadata:
            # naziv fajla = arxiv id sa _ umesto /
            filename = paper_id.replace("/", "_") + ".json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"  Sacuvano: {filename} — {metadata.get('title', '')[:60]}")
        else:
            print(f"  Preskoceno: {paper_id}")

        time.sleep(1)  # arXiv API rate limit

    print(f"\nGotovo! Sacuvano u: {output_dir}/")


if __name__ == "__main__":
    main()