import json
import os

EXTRACTED_DIR = "./data/extracted_2"

# arxiv_id -> tačna godina iz arxiv ID-a (YYMM format)
CORRECT_YEARS = {
    "2303.18223": "2023",
    "2404.10981": "2024", 
    "2204.03954": "2022",
    "2408.08073": "2024",
    "2305.14842": "2023",
    "2409.09989": "2024",
    "2503.20227": "2025",
    "1810.04805": "2018",  # BERT — arxiv Oct 2018
    "1904.08067": "2019",
    "1907.11692": "2019",
    "1908.10084": "2019",
}

for arxiv_id, correct_year in CORRECT_YEARS.items():
    filepath = os.path.join(EXTRACTED_DIR, f"{arxiv_id}_extraction.json")
    if not os.path.exists(filepath):
        print(f"NOT FOUND: {filepath}")
        continue
    
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    
    old_year = data.get("year")
    data["year"] = correct_year
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"{arxiv_id}: {old_year} → {correct_year}")

print("\nDone!")