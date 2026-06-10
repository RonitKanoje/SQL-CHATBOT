from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    user_message: str
    messages: Annotated[list[BaseMessage], add_messages]
    is_safe: bool
    guardrail_reason: str
    normalized_question: str
    retrieved_docs: list[Any]
    join_candidates: list[dict]
    context: str
    sql: str
    sql_valid: bool
    sql_retry_count: int
    validation_error: str
    sql_result: list[dict]
    error: str
    answer: str
    request_started_at: float
    timings: dict[str, float]
    sql_cache_hit: bool
    result_cache_hit: bool
