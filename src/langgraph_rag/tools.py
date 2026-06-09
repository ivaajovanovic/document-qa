import os
import json
import logging
from langchain_core.tools import tool
from src.langgraph_rag.retriever_utils import get_retriever, load_config

logger = logging.getLogger(__name__)

EXTRACTED_DIR = "./data/extracted_2"
DEFAULT_CONFIG_ID = "config_256_k10_rrf60"


def load_all_extracted() -> list[dict]:
    """Load all extracted paper metadata."""
    papers = []
    for filename in os.listdir(EXTRACTED_DIR):
        if not filename.endswith("_extraction.json"):
            continue
        arxiv_id = filename.replace("_extraction.json", "")
        try:
            with open(os.path.join(EXTRACTED_DIR, filename), encoding="utf-8") as f:
                data = json.load(f)
                data["arxiv_id"] = arxiv_id
                papers.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
    return papers


def search_with_filter(query: str, arxiv_ids: list[str], config_id: str = DEFAULT_CONFIG_ID) -> list[dict]:
    """Search FAISS but only return chunks from filtered arxiv_ids."""
    try:
        config = load_config(config_id)
        retriever = get_retriever(config)
        results = retriever.search(query, top_k=config["top_k"], use_rrf=True)
        hybrid = results["hybrid"]
        return [c for c in hybrid if c["metadata"].get("arxiv_id") in arxiv_ids]
    except Exception as e:
        logger.error(f"search_with_filter failed: {e}")
        return []


def format_chunks(chunks: list[dict]) -> str:
    """Format chunks as readable string."""
    if not chunks:
        return "No relevant content found."
    results = []
    for c in chunks[:5]:
        results.append(
            f"[{c['metadata'].get('title')} ({c['metadata'].get('arxiv_id')}), "
            f"page {c['metadata'].get('page_num')}]\n{c['text']}"
        )
    return "\n\n".join(results)


@tool
def search_papers(query: str) -> str:
    """
    Search academic papers by content using RAG retrieval.
    Use this for general questions about paper content without any filters.

    Args:
        query: The search query about paper content.

    Example: search_papers("how does self-attention work")
    """
    try:
        config = load_config(DEFAULT_CONFIG_ID)
        retriever = get_retriever(config)
        results = retriever.search(query, top_k=config["top_k"], use_rrf=True)
        chunks = results["hybrid"]
        return format_chunks(chunks)
    except Exception as e:
        logger.error(f"search_papers failed: {e}")
        return f"Error searching papers: {e}"


@tool
def search_papers_by_year(query: str, before: str, after: str) -> str:
    """
    Search academic papers filtered by publication year.

    Args:
        query: The search query about paper content.
        before: Only include papers published before this year. Use "0" for no upper limit.
        after: Only include papers published after this year. Use "0" for no lower limit.

    Example: To find papers before 2018, use before="2018", after="0".
    """
    try:
        before_int = int(before) if before else 0
        after_int = int(after) if after else 0
    except (ValueError, TypeError):
        before_int = 0
        after_int = 0

    try:
        papers = load_all_extracted()

        filtered = []
        for p in papers:
            try:
                year = int(p.get("year", 0))
            except (ValueError, TypeError):
                continue
            if before_int > 0 and year >= before_int:
                continue
            if after_int > 0 and year <= after_int:
                continue
            filtered.append(p["arxiv_id"])

        if not filtered:
            return f"No papers found matching year filter (before={before}, after={after})."

        chunks = search_with_filter(query, filtered)
        paper_list = ", ".join(filtered)
        return f"Papers matching year filter: {paper_list}\n\n{format_chunks(chunks)}"

    except Exception as e:
        logger.error(f"search_papers_by_year failed: {e}")
        return f"Error searching papers by year: {e}"


@tool
def search_papers_by_company(query: str, company: str) -> str:
    """
    Search academic papers from a specific company or institution.

    Args:
        query: The search query about paper content.
        company: Company or institution name to filter by (e.g. 'Google', 'MIT', 'Facebook').

    Example: To find Google papers about transformers, use query='transformer', company='Google'.
    """
    try:
        papers = load_all_extracted()

        filtered = []
        for p in papers:
            companies = [c.lower() for c in p.get("companies", [])]
            if any(company.lower() in c for c in companies):
                filtered.append(p["arxiv_id"])

        if not filtered:
            return f"No papers found from company: {company}"

        chunks = search_with_filter(query, filtered)
        paper_list = ", ".join(filtered)
        return f"Papers from {company}: {paper_list}\n\n{format_chunks(chunks)}"

    except Exception as e:
        logger.error(f"search_papers_by_company failed: {e}")
        return f"Error searching papers by company: {e}"


@tool
def search_papers_by_author(query: str, author: str) -> str:
    """
    Search academic papers by a specific author.

    Args:
        query: The search query about paper content.
        author: Author name or partial name to filter by.

    Example: To find Vaswani's papers about attention, use query='attention', author='Vaswani'.
    """
    try:
        papers = load_all_extracted()

        filtered = []
        for p in papers:
            authors = [a.lower() for a in p.get("authors", [])]
            if any(author.lower() in a for a in authors):
                filtered.append(p["arxiv_id"])

        if not filtered:
            return f"No papers found by author: {author}"

        chunks = search_with_filter(query, filtered)
        paper_list = ", ".join(filtered)
        return f"Papers by {author}: {paper_list}\n\n{format_chunks(chunks)}"

    except Exception as e:
        logger.error(f"search_papers_by_author failed: {e}")
        return f"Error searching papers by author: {e}"


@tool
def list_papers_metadata(company: str, year_before: str, year_after: str) -> str:
    """
    List papers matching metadata filters without doing RAG search.
    Useful to see which papers are available before searching.

    Args:
        company: Filter by company. Use empty string for no filter.
        year_before: Filter papers before this year. Use "0" for no filter.
        year_after: Filter papers after this year. Use "0" for no filter.
    """
    try:
        year_before_int = int(year_before) if year_before else 0
        year_after_int = int(year_after) if year_after else 0
    except (ValueError, TypeError):
        year_before_int = 0
        year_after_int = 0

    try:
        papers = load_all_extracted()

        filtered = []
        for p in papers:
            try:
                year = int(p.get("year", 0))
            except (ValueError, TypeError):
                continue
            if year_before_int > 0 and year >= year_before_int:
                continue
            if year_after_int > 0 and year <= year_after_int:
                continue
            if company:
                companies = [c.lower() for c in p.get("companies", [])]
                if not any(company.lower() in c for c in companies):
                    continue
            filtered.append(p)

        if not filtered:
            return "No papers found matching filters."

        results = []
        for p in filtered:
            results.append(
                f"- {p['title']} ({p['arxiv_id']}, {p['year']}) "
                f"| Companies: {', '.join(p.get('companies', []))}"
            )
        return "\n".join(results)

    except Exception as e:
        logger.error(f"list_papers_metadata failed: {e}")
        return f"Error listing papers: {e}"


@tool
def get_paper_authors(arxiv_id: str) -> str:
    """
    Get all authors of a specific paper by arxiv ID.
    Use this when asked about authors of a specific paper.

    Args:
        arxiv_id: The arxiv ID of the paper (e.g. '1706.03762').

    Example: get_paper_authors("1706.03762")
    """
    try:
        papers = load_all_extracted()
        for p in papers:
            if p["arxiv_id"] == arxiv_id:
                authors = ", ".join(p.get("authors", []))
                return f"Authors of '{p['title']}' ({arxiv_id}): {authors}"
        return f"Paper {arxiv_id} not found."
    except Exception as e:
        logger.error(f"get_paper_authors failed: {e}")
        return f"Error getting authors: {e}"


TOOLS = [
    search_papers,
    search_papers_by_year,
    search_papers_by_company,
    search_papers_by_author,
    list_papers_metadata,
    get_paper_authors,
]