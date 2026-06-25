"""Pydantic models for API request/response validation."""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """User message request to the RAG chat pipeline.
    
    Attributes:
        thread_id: Session identifier for conversation persistence.
        message: User query to be processed by the RAG system.
        config_id: Retrieval configuration ID (e.g., 'config_multimodal_k10_rrf60').
                   Defaults to multimodal config.
    """
    thread_id: str
    message: str
    config_id: Optional[str] = "config_multimodal_k10_rrf60"


class ChatResponse(BaseModel):
    """Response from the RAG chat endpoint.
    
    Attributes:
        answer: Generated answer from the LLM based on retrieved context.
        thread_id: Session identifier for conversation tracking.
        images: Base64-encoded images from retrieved figure chunks. Empty list if no figures found.
    """
    answer: str
    thread_id: str
    images: Optional[list[str]] = []


class NewSessionResponse(BaseModel):
    """Response after creating a new chat session.
    
    Attributes:
        thread_id: Unique identifier for the new session.
    """
    thread_id: str