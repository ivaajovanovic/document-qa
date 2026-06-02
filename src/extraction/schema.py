from pydantic import BaseModel


class PaperExtraction(BaseModel):
    title: str
    title_evidence: str
    authors: list[str]
    authors_evidence: str
    companies: list[str]
    year: str
    year_evidence: str
    primary_topic: str
    keywords: list[str]
    methodology: str
    methodology_evidence: str