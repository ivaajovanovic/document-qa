import logging
from src.extraction.client import GroqClient
from src.extraction.prompts import EXTRACTION_PROMPT
from src.extraction.schema import PaperExtraction

logger = logging.getLogger(__name__)


def extract_paper_info(text: str, client: GroqClient) -> PaperExtraction | None:
    """
    Extract structured paper metadata from raw text via LLM.

    The flow is straightforward:
    1) take paper text,
    2) place it into the extraction prompt,
    3) parse model output into ``PaperExtraction``.

    Long papers are truncated so request size stays predictable.

    Args:
        text: Full paper text assembled from chunk content.
        client: Configured Groq client that supports structured output.

    Returns:
        ``PaperExtraction`` on success, otherwise ``None``.
    """
    # Keep only the leading segment to reduce token usage and avoid truncation
    # by the provider on very long papers.
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])

    try:
        return client.complete_structured(prompt, PaperExtraction)

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None