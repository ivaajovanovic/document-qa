import os
import re
import time
import base64
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

FIGURES_DIR = "./data/figures"

from dotenv import load_dotenv
load_dotenv()


def clean_text(text: str) -> str:
    text = re.sub(r'arXiv:\S+', '', text)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'[^\w\s.,;:!?()\-\'\"]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _encode_image_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return None


def _call_vision_llm(client, image_data: str, media_type: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def _generate_figure_description(image_path: str, caption: str) -> str:
    """
    Use multimodal LLM to generate description of figure.
    Uses llama-4-scout on Groq (free tier).
    Handles rate limits with automatic wait and retry.
    """
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        image_data = _encode_image_base64(image_path)
        if not image_data:
            return caption or "Figure from academic paper."

        ext = image_path.rsplit(".", 1)[-1].lower()
        media_type = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "gif", "webp"] else "image/png"

        prompt = "Describe this academic figure concisely. Focus on: what it shows, key components, and main takeaway. Keep it under 100 words."
        if caption:
            prompt += f" The caption reads: '{caption}'"

        try:
            return _call_vision_llm(client, image_data, media_type, prompt)

        except Exception as e:
            error_str = str(e)

            if "429" in error_str:
                wait_match = re.search(r'try again in (\d+)m([\d.]+)s', error_str)
                if wait_match:
                    wait = int(wait_match.group(1)) * 60 + float(wait_match.group(2))
                    logger.info(f"Rate limit hit for {image_path}, waiting {wait:.0f}s...")
                    time.sleep(wait + 5)
                    try:
                        return _call_vision_llm(client, image_data, media_type, prompt)
                    except Exception as retry_e:
                        logger.warning(f"Vision LLM retry failed for {image_path}: {retry_e}")
                        return caption or "Figure from academic paper."
                else:
                    # nema info koliko da čeka — preskoči odmah
                    logger.warning(f"Rate limit hit, skipping {image_path}")
                    return caption or "Figure from academic paper."

            logger.warning(f"Vision LLM failed for {image_path}: {e}")
            return caption or "Figure from academic paper."

    except Exception as e:
        logger.warning(f"Vision LLM setup failed for {image_path}: {e}")
        return caption or "Figure from academic paper."


def chunk_text_multimodal(pages: list[dict], metadata: dict, generate_descriptions: bool = True) -> list[dict]:
    """
    Chunk text and figures from PDF pages.

    Args:
        pages: Output from load_pdf() — list of dicts with page_num, text, images.
        metadata: Paper metadata with title, authors, year, arxiv_id.
        generate_descriptions: If True, use Vision LLM for figure descriptions.

    Returns:
        List of chunks with chunk_type "text" or "figure".
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=28,
    )

    chunks = []
    chunk_index = 0
    char_position = 0
    figure_index = 0

    for page in pages:
        # --- TEXT CHUNKS ---
        cleaned = clean_text(page["text"])
        if cleaned:
            page_chunks = splitter.split_text(cleaned)
            for chunk in page_chunks:
                if not chunk.strip():
                    continue
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_type": "text",
                        "chunk_index": chunk_index,
                        "page_num": page["page_num"],
                        "char_start": char_position,
                        "char_end": char_position + len(chunk),
                    }
                })
                char_position += len(chunk)
                chunk_index += 1

        # --- FIGURE CHUNKS ---
        for img in page.get("images", []):
            image_path = img.get("image_path")
            caption = img.get("caption", "")
            figure_id = f"fig_{figure_index}"

            if image_path and generate_descriptions:
                description = _generate_figure_description(image_path, caption)
            else:
                description = caption or "Figure from academic paper."

            chunk_text = f"{caption} {description}".strip() if caption else description

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_type": "figure",
                    "chunk_index": chunk_index,
                    "page_num": page["page_num"],
                    "figure_id": figure_id,
                    "image_path": image_path,
                    "caption": caption,
                    "llm_description": description,
                    "char_start": None,
                    "char_end": None,
                }
            })
            chunk_index += 1
            figure_index += 1

    return chunks