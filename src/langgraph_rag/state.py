from typing import Annotated, Optional
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


class InputState(TypedDict):
    """Input state — only messages for Studio Chat compatibility."""
    messages: Annotated[list, add_messages]


class RAGState(InputState):
    """Full internal state."""
    config_id: NotRequired[str]
    retrieved_chunks: NotRequired[Optional[dict]]
    answer: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]