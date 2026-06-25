import uuid
import base64
import traceback
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.langgraph_rag.graph import rag_graph
from src.langgraph_rag.chat_service import (
    DEFAULT_CONFIG,
    create_initial_state,
    ask_chatbot,
)
from src.langgraph_rag.session_manager import (
    create_session,
    update_session,
    get_all_sessions,
    delete_session,
    save_message_images,
    get_message_images,
)
from src.api.schemas import ChatRequest, ChatResponse, NewSessionResponse
from src.api.dependencies import get_configs

router = APIRouter()

SESSION_STATES = {}


def _encode_image(image_path: str) -> str | None:
    """Encode image file to base64 data URL for frontend display.
    
    Args:
        image_path: Local file path to the image.
        
    Returns:
        Data URL string (e.g., 'data:image/png;base64,...') or None if encoding fails.
    """
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = image_path.rsplit(".", 1)[-1].lower()
            return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"[IMAGES] Failed to encode {image_path}: {e}")
        return None


def _extract_images_from_state(state: dict) -> tuple[list[str], list[str]]:
    """
    Search FAISS for figure chunks matching the last question.
    Returns (base64_images, image_paths) tuple.
    base64_images — for immediate display in frontend.
    image_paths — for storage and future re-encoding.
    In production images would be URLs from object storage (S3, GCS etc.).
    """
    from src.langgraph_rag.retriever_utils import get_retriever, load_config

    images = []
    image_paths = []
    config_id = state.get("config_id", "config_multimodal_k10_rrf60")

    question = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    if not question:
        return [], []

    try:
        config = load_config(config_id)
        retriever = get_retriever(config)
        results = retriever.search(question, top_k=config["top_k"], use_rrf=True)
        chunks = results.get("hybrid", [])

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if metadata.get("chunk_type") != "figure":
                continue
            image_path = metadata.get("image_path")
            if not image_path:
                continue
            encoded = _encode_image(image_path)
            if encoded:
                images.append(encoded)
                image_paths.append(image_path)

    except Exception as e:
        print(f"[IMAGES] Search failed: {e}")

    return images, image_paths


@router.get("/sessions")
def list_sessions():
    """List all active chat sessions ordered by most recent first.
    
    Returns:
        List of session metadata dicts with thread_id, name, created_at, last_updated.
    """
    return get_all_sessions()


@router.post("/sessions/new", response_model=NewSessionResponse)
def new_session():
    """Create a new chat session with a unique thread_id.
    
    Returns:
        NewSessionResponse with the generated thread_id for use in future requests.
    """
    thread_id = str(uuid.uuid4())
    SESSION_STATES[thread_id] = create_initial_state(DEFAULT_CONFIG)
    return {"thread_id": thread_id}


@router.delete("/sessions/{thread_id}")
def remove_session(thread_id: str):
    """Delete a chat session and its conversation history.
    
    Args:
        thread_id: The session ID to delete.
        
    Returns:
        Status confirmation dict.
    """
    delete_session(thread_id)
    SESSION_STATES.pop(thread_id, None)
    return {"status": "deleted"}


@router.get("/configs")
def list_configs():
    """List all available retrieval configurations.
    
    Returns:
        List of config dicts with id and description for frontend selection.
    """
    return [{"id": c["id"], "description": c["description"]} for c in get_configs()]


@router.get("/sessions/{thread_id}/history")
def get_history(thread_id: str):
    """Retrieve full conversation history for a session.
    
    Fetches all messages and associated images for a given thread_id.
    Restores from LangGraph checkpoints if not in memory.
    
    Args:
        thread_id: Session identifier.
        
    Returns:
        List of message dicts with role ('user' or 'assistant'), content, and images.
    """
    try:
        state = SESSION_STATES.get(thread_id)

        if state is None:
            graph_config = {"configurable": {"thread_id": thread_id}}
            graph_state = rag_graph.get_state(graph_config)
            if not graph_state or not graph_state.values:
                return []
            state = graph_state.values

        messages = state.get("messages", [])
        # Load image paths from session_manager
        message_images = get_message_images(thread_id)

        history = []
        assistant_index = 0

        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content, "images": []})
            elif isinstance(msg, AIMessage) and msg.content:
                # Re-encode images from saved paths
                paths = message_images.get(str(assistant_index), [])
                images = [enc for p in paths if (enc := _encode_image(p))]
                history.append({
                    "role": "assistant",
                    "content": msg.content,
                    "images": images,
                })
                assistant_index += 1

        return history

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Process a user query through the RAG chat pipeline.
    
    Runs the LangGraph workflow: query decomposition → agent/tool selection →
    retrieval → LLM answer generation → figure extraction. Persists session
    state by thread_id for multi-turn conversations.
    
    Args:
        req: ChatRequest with thread_id, message, and optional config_id.
        
    Returns:
        ChatResponse with generated answer, thread_id, and base64 images from
        retrieved figure chunks (if multimodal config).
        
    Raises:
        HTTPException: If RAG pipeline fails (500 status).
    """
    try:
        thread_id = req.thread_id or str(uuid.uuid4())
        config_id = req.config_id or DEFAULT_CONFIG

        state = SESSION_STATES.get(thread_id)
        if state is None:
            state = create_initial_state(config_id)

        state["config_id"] = config_id

        result = ask_chatbot(
            message=req.message,
            state=state,
            thread_id=thread_id,
            config_id=config_id,
        )

        SESSION_STATES[thread_id] = result["state"]

        images, image_paths = _extract_images_from_state(result["state"])

        sessions = get_all_sessions()
        existing_thread_ids = [s["thread_id"] for s in sessions]

        if thread_id not in existing_thread_ids:
            create_session(thread_id, req.message)
        else:
            update_session(thread_id)

        # Save image paths using the assistant message index
        if image_paths:
            # Number of assistant messages equals number of AI messages in state
            ai_count = sum(
                1 for m in result["state"].get("messages", [])
                if isinstance(m, AIMessage) and m.content
            )
            save_message_images(thread_id, ai_count - 1, image_paths)

        return {
            "answer": result.get("answer", "No answer generated."),
            "thread_id": thread_id,
            "images": images,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))