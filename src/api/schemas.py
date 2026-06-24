from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    thread_id: str
    message: str
    config_id: Optional[str] = "config_multimodal_k10_rrf60"

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    images: Optional[list[str]] = []  # base64 encoded images from figure chunks

class NewSessionResponse(BaseModel):
    thread_id: str