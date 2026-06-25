# Document Question Answering System

Hybrid RAG system for question answering over PDF documents with multimodal retrieval.

## Quick Start

1. Open the project root:
```bash
cd document-qa
```

2. Install dependencies:
```bash
poetry install
```

3. Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Prepare the dataset:
```bash
poetry run python scripts/fetch_metadata.py
```

5. Download arXiv PDFs into `data/raw/arxiv_papers/`.

6. Run multimodal preprocessing:
```bash
poetry run python src/ingestion/run_multimodal.py
```

7. Build search indexes:
```bash
poetry run python scripts/build_indexes.py
poetry run python scripts/build_multimodal_index.py
```

8. Start the backend API:
```bash
poetry run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

9. Start the frontend:
```bash
poetry run streamlit run frontend/app.py
```

10. Open in a browser:
- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

## Project Structure

```
.
├── data/
│   ├── processed_256/
│   ├── processed_128/
│   ├── processed_multimodal/
│   ├── extracted/
│   ├── extracted_2/
│   └── figures/
├── experiments/
│   └── configs_langgraph.json
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── components/
│   └── styles.css
├── scripts/
│   ├── build_indexes.py
│   ├── build_multimodal_index.py
│   ├── fetch_metadata.py
│   └── generate_qa.py
├── src/
│   ├── api/
│   ├── langgraph_rag/
│   ├── rag/
│   ├── ingestion/
│   └── extraction/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Backend

### `src/api/`
- `main.py` — FastAPI application
- `routes.py` — API endpoints
- `schemas.py` — Pydantic request/response models
- `dependencies.py` — shared configuration and loaders

### `src/langgraph_rag/`
- `graph.py` — LangGraph workflow definition
- `nodes.py` — workflow node functions
- `tools.py` — tool handlers
- `retriever_utils.py` — retriever selection by config
- `state.py` — workflow state structure

### `src/rag/`
- `retriever.py` — base text retriever
- `retriever_multimodal.py` — multimodal retriever with figure boosting
- `embedder.py` — embedder setup
- `generator.py` — answer generation

## Frontend

- `frontend/app.py` — Streamlit UI
- `frontend/api_client.py` — backend API client
- `frontend/components/chat.py` — chat interface
- `frontend/components/sidebar.py` — session and config panel
- `frontend/styles.css` — custom styles

## Scripts

- `scripts/build_indexes.py` — build text retrieval indexes
- `scripts/build_multimodal_index.py` — build multimodal index
- `scripts/fetch_metadata.py` — fetch metadata for papers
- `scripts/generate_qa.py` — generate Q&A pairs

## Configuration

All retrieval configs are stored in `experiments/configs_langgraph.json`.

### Key configs
- `config_multimodal_k10_rrf60` — multimodal retrieval (text + figures)
- `config_256_k10_rrf60` — text-only retrieval, chunk 256
- `config_128_k5_rrf60` — text-only retrieval, chunk 128

## API Endpoints

- `POST /chat` — send a question
- `GET /configs` — list available configurations
- `GET /sessions` — list sessions
- `POST /sessions/new` — create a new session
- `DELETE /sessions/{thread_id}` — delete a session
- `GET /history/{thread_id}` — retrieve chat history

## Data and preprocessing

This project is built from arXiv papers and metadata. The pipeline converts PDFs into retrieval-ready text and figure chunks, then embeds them for RAG search.

### Required inputs

- `data/raw/arxiv_papers/<arxiv_id>.pdf` — raw PDF content
- `data/raw/arxiv_papers/<arxiv_id>.json` — metadata used during chunking and retrieval

### Processing flow

1. **Load the PDF**
   - `src/ingestion/pdf_loader_multimodal.py` opens each PDF and extracts page text.
   - It detects raster and vector figures, finds candidate captions, and writes extracted images to `data/figures/<arxiv_id>/`.

2. **Chunk the content**
   - `src/ingestion/chunker_multimodal.py` cleans text and splits it with `RecursiveCharacterTextSplitter`.
   - Text is chunked at ~256 characters with overlap to preserve context.
   - Figure content is indexed as separate chunks using the caption plus an optional vision-generated description.

3. **Embed the chunks**
   - `src/rag/embedder.py` uses `OllamaEmbeddings` from `langchain_ollama`.
   - The `embed_chunks()` helper converts chunk text into vector embeddings and attaches them to each chunk dict.

4. **Build retrieval indexes**
   - `scripts/build_indexes.py` loads processed chunk files and builds indexes according to `experiments/configs_langgraph.json`.
   - `scripts/build_multimodal_index.py` specifically loads `data/processed_multimodal/` and builds a multimodal FAISS/BM25 index.

## Notes

- The multimodal pipeline is designed to index both text and figure content.
- Figure descriptions may use Groq vision if `GROQ_API_KEY` is set.
- Index files are stored in `data/cache/` by the retriever code.

## Environment variables

```env
GROQ_API_KEY=your_groq_api_key_here
EMBED_MODEL=mxbai-embed-large
```


