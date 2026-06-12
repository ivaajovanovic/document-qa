from typing import Annotated, Optional
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class InputState(TypedDict):
    messages: Annotated[list, add_messages]


class RAGState(InputState):
    config_id: NotRequired[str]
    retrieved_chunks: NotRequired[Optional[dict]]
    answer: NotRequired[Optional[str]]
    error: NotRequired[Optional[str]]
    sub_queries: NotRequired[Optional[list[str]]]