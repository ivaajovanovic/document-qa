"""FastAPI application setup for the Document QA backend.

Provides REST endpoints for:
- Chat queries with RAG retrieval
- Session management
- Configuration selection
- Retrieved image extraction from figure chunks
"""

from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Document QA API")
app.include_router(router)