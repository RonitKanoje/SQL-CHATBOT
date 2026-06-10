import copy
import json
import re
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langsmith import traceable
from database.mysql import clean_sql, execute_read_query, validate_read_query
from graph.state import ChatState
from llm.gemini import generate_sql
from prompts.sql_prompt import build_sql_generation_prompt
from retrieval.context_builder import build_context
from retrieval.join_expander import expand_connected_tables
from retrieval.retriever import retrieve_embed


BLOCKED_USER_INTENT = re.compile(
    r"\b(update|delete|create|insert|drop|alter|truncate|replace|remove|modify)\b",
    re.IGNORECASE,
)

MAX_SQL_RETRIES = 2
QUESTION_SQL_CACHE: dict[str, str] = {}
QUERY_RESULT_CACHE: dict[str, list[dict]] = {}


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _timings_with(state: ChatState, name: str, elapsed: float) -> dict[str, float]:
    timings = dict(state.get("timings") or {})
    timings[name] = timings.get(name, 0.0) + elapsed
    return timings


def _print_benchmark(state: ChatState, timings: dict[str, float]) -> None:
    started_at = state.get("request_started_at")
    total_time = time.perf_counter() - started_at if started_at else sum(timings.values())
    print(
        "request_benchmark\n"
        f"retrieve_time={timings.get('retrieve_time', 0.0):.3f}s\n"
        f"generate_sql_time={timings.get('generate_sql_time', 0.0):.3f}s\n"
        f"validate_time={timings.get('validate_time', 0.0):.3f}s\n"
        f"execute_time={timings.get('execute_time', 0.0):.3f}s\n"
        f"explain_time={timings.get('explain_time', 0.0):.3f}s\n"
        f"total_time={total_time:.3f}s"
    )


def _chat_history(messages: list[BaseMessage] | None, max_messages: int = 8) -> str:
    if not messages:
        return "No previous conversation."

    lines = []
    for message in messages[-max_messages:]:
        role = "User" if message.type == "human" else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


@traceable(name="Read Only Guardrail", run_type="llm", tags=["guardrail", "validation"])
def guardrail_node(state: ChatState) -> dict:
    started_at = time.perf_counter()
    question = state["user_message"]
    is_safe = BLOCKED_USER_INTENT.search(question) is None   ###
    base_state = {
        "normalized_question": _normalize_question(question),
        "request_started_at": started_at,
        "timings": {},
        "sql_cache_hit": False,
        "result_cache_hit": False,
    }
    if is_safe:
        return {
            **base_state,
            "is_safe": True,
            "sql": "",
            "sql_valid": False,
            "sql_retry_count": 0,
            "validation_error": "",
            "error": "",
            "sql_result": [],
            "messages": [HumanMessage(content=question)],
        }

    return {
        **base_state,
        "is_safe": False,
        "guardrail_reason": "Only read-only database questions are allowed.",
        "sql": "",
        "sql_valid": False,
        "sql_retry_count": 0,
        "validation_error": "",
        "error": "",
        "sql_result": [],
        "messages": [HumanMessage(content=question)],
    }


def guardrail_route(state: ChatState) -> str:
    return "allowed" if state.get("is_safe") else "blocked"


@traceable(name="Blocked Response", run_type="llm", tags=["guardrail", "response"])
def blocked_node(state: ChatState) -> dict:
    answer = "I can help with read-only database questions, but I cannot perform or generate update, delete, create, insert, drop, or alter operations."
    _print_benchmark(state, state.get("timings") or {})
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


@traceable(name="Retrieve Schema", run_type="retriever", tags=["retrieval", "embeddings"])
def retrieval_node(state: ChatState) -> dict:
    started_at = time.perf_counter()
    docs = retrieve_embed(state["user_message"])
    expanded_docs, join_candidates = expand_connected_tables(state["user_message"], docs)
    return {
        "retrieved_docs": expanded_docs,
        "join_candidates": join_candidates,
        "timings": _timings_with(state, "retrieve_time", time.perf_counter() - started_at),
    }


@traceable(name="Build Context", run_type="retriever", tags=["context", "sql"])
def context_node(state: ChatState) -> dict:
    context = build_context(
        state.get("retrieved_docs", []),
        state.get("join_candidates", []),
    )
    return {"context": context}


@traceable(name="Generate SQL", run_type="llm", tags=["sql", "generation"])
def generate_sql_node(state: ChatState) -> dict:
    started_at = time.perf_counter()
    normalized_question = state.get("normalized_question") or _normalize_question(state["user_message"])
    validation_feedback = ""
    if state.get("validation_error"):
        validation_feedback = (
            f"SQL failed validation: {state['validation_error']}\n"
            f"Previous SQL: {state.get('sql', '')}\n"
            "Generate a corrected read-only MySQL query."
        )

    if not validation_feedback:
        cached_sql = QUESTION_SQL_CACHE.get(normalized_question)
        if cached_sql:
            return {
                "sql": cached_sql,
                "sql_valid": False,
                "error": "",
                "normalized_question": normalized_question,
                "sql_cache_hit": True,
                "timings": _timings_with(
                    state,
                    "generate_sql_time",
                    time.perf_counter() - started_at,
                ),
            }

    prompt = build_sql_generation_prompt( #####
        question=state["user_message"],
        context=state.get("context", ""),
        chat_history=_chat_history(state.get("messages")),
        validation_feedback=validation_feedback,
    )
    sql = clean_sql(generate_sql(prompt))

    return {
        "sql": sql,
        "sql_valid": False,
        "error": "",
        "normalized_question": normalized_question,
        "sql_cache_hit": False,
        "timings": _timings_with(
            state,
            "generate_sql_time",
            time.perf_counter() - started_at,
        ),
    }


@traceable(name="Validate SQL", run_type="tool", tags=["validation", "sql"])
def validate_sql_node(state: ChatState) -> dict:
    started_at = time.perf_counter()
    is_valid, validation_error = validate_read_query(state.get("sql", ""))
    timings = _timings_with(state, "validate_time", time.perf_counter() - started_at)
    if is_valid:
        normalized_question = state.get("normalized_question") or _normalize_question(state["user_message"])
        QUESTION_SQL_CACHE[normalized_question] = state.get("sql", "")
        return {
            "sql_valid": True,
            "validation_error": "",
            "error": "",
            "normalized_question": normalized_question,
            "timings": timings,
        }

    retry_count = state.get("sql_retry_count", 0)
    if retry_count < MAX_SQL_RETRIES:
        return {
            "sql_valid": False,
            "validation_error": validation_error,
            "sql_retry_count": retry_count + 1,
            "error": "",
            "timings": timings,
        }

    return {
        "sql_valid": False,
        "validation_error": validation_error,
        "error": f"SQL validation failed after {MAX_SQL_RETRIES} retries: {validation_error}",
        "timings": timings,
    }


def validate_sql_route(state: ChatState) -> str:
    if state.get("sql_valid"):
        return "valid"
    if state.get("sql_retry_count", 0) <= MAX_SQL_RETRIES and not state.get("error"):
        return "retry"
    return "failed"


@traceable(name="SQL Failure Response", run_type="llm", tags=["failure", "response"])
def sql_failure_node(state: ChatState) -> dict:
    answer = (
        "I could not generate a valid read-only SQL query for that request after "
        f"{MAX_SQL_RETRIES} retries. Last validation error: "
        f"{state.get('validation_error', 'Unknown validation error.')}"
    )
    _print_benchmark(state, state.get("timings") or {})
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


@traceable(name="Execute SQL", run_type="tool", tags=["database", "execution"])
def execute_sql_node(state: ChatState) -> dict:
    started_at = time.perf_counter()
    if state.get("error"):
        return {
            "sql_result": [],
            "timings": _timings_with(state, "execute_time", time.perf_counter() - started_at),
        }

    sql_key = clean_sql(state["sql"])
    cached_result = QUERY_RESULT_CACHE.get(sql_key)
    if cached_result is not None:
        return {
            "sql_result": copy.deepcopy(cached_result),
            "result_cache_hit": True,
            "timings": _timings_with(state, "execute_time", time.perf_counter() - started_at),
        }

    try:
        result = execute_read_query(sql_key)
        QUERY_RESULT_CACHE[sql_key] = copy.deepcopy(result)
        return {
            "sql_result": result,
            "result_cache_hit": False,
            "timings": _timings_with(state, "execute_time", time.perf_counter() - started_at),
        }
    except Exception as exc:   
        return {
            "sql_result": [],
            "error": str(exc),
            "timings": _timings_with(state, "execute_time", time.perf_counter() - started_at),
        }


@traceable(name="Explain SQL Result", run_type="llm", tags=["explanation", "response"])
def explain_result_node(state: ChatState) -> dict:
    started_at = time.perf_counter()
    if state.get("error"):
        answer = f"Execution error: {state['error']}"
    elif not state.get("sql_result"):
        answer = "No matching records were found."
    else:
        answer = json.dumps(
            state.get("sql_result", []),
            default=str,
            indent=2,
        )

    timings = _timings_with(state, "explain_time", time.perf_counter() - started_at)
    _print_benchmark(state, timings)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "timings": timings,
    }
