import logging
from src.extraction.client import GroqClient
from src.extraction.prompts import EXTRACTION_PROMPT
from src.extraction.schema import PaperExtraction

logger = logging.getLogger(__name__)


def extract_paper_info(text: str, client: GroqClient) -> PaperExtraction | None:
    """
    Extract structured information from paper text using structured output.

    Args:
        text: Full paper text.
        client: GroqClient instance.

    Returns:
        PaperExtraction object if successful, None if extraction fails.
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])

    try:
        return client.complete_structured(prompt, PaperExtraction)

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None