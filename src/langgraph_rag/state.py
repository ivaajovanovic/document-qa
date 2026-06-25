from typing import Annotated, Optional
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class InputState(TypedDict):
    """Minimal graph input: just the running message list."""
    messages: Annotated[list, add_messages]


class RAGState(InputState):
    """Extended state shared across all graph nodes.

    `NotRequired` fields are filled progressively as the pipeline runs.
    """
    config_id: NotRequired[str]
    retrieved_chunks: NotRequired[Optional[dict]]
    answer: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    sub_queries: NotRequired[Optional[list[str]]]