import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langsmith import traceable

from database.mysql import clean_sql, execute_read_query, validate_read_query
from graph.state import ChatState
from llm.gemini import explain_result, generate_sql
from prompts.sql_prompt import build_result_explanation_prompt, build_sql_generation_prompt
from retrieval.context_builder import build_context
from retrieval.join_expander import expand_connected_tables
from retrieval.retriever import retrieve_embed


BLOCKED_USER_INTENT = re.compile(
    r"\b(update|delete|create|insert|drop|alter|truncate|replace|remove|modify)\b",
    re.IGNORECASE,
)
MAX_SQL_RETRIES = 2


def _chat_history(messages: list[BaseMessage] | None, max_messages: int = 8) -> str:
    if not messages:
        return "No previous conversation."

    lines = []
    for message in messages[-max_messages:]:
        role = "User" if message.type == "human" else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


@traceable(name="Read Only Guardrail")
def guardrail_node(state: ChatState) -> dict:
    question = state["user_message"]
    is_safe = BLOCKED_USER_INTENT.search(question) is None
    if is_safe:
        return {
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


@traceable(name="Blocked Response")
def blocked_node(state: ChatState) -> dict:
    answer = (
        "I can help with read-only database questions, but I cannot perform "
        "or generate update, delete, create, insert, drop, or alter operations."
    )
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


@traceable(name="Retrieve Schema")
def retrieval_node(state: ChatState) -> dict:
    docs = retrieve_embed(state["user_message"])
    expanded_docs, join_candidates = expand_connected_tables(state["user_message"], docs)
    return {
        "retrieved_docs": expanded_docs,
        "join_candidates": join_candidates,
    }


@traceable(name="Build Context")
def context_node(state: ChatState) -> dict:
    context = build_context(
        state.get("retrieved_docs", []),
        state.get("join_candidates", []),
    )
    return {"context": context}


@traceable(name="Generate SQL")
def generate_sql_node(state: ChatState) -> dict:
    validation_feedback = ""
    if state.get("validation_error"):
        validation_feedback = (
            f"SQL failed validation: {state['validation_error']}\n"
            f"Previous SQL: {state.get('sql', '')}\n"
            "Generate a corrected read-only MySQL query."
        )

    prompt = build_sql_generation_prompt(
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
    }


@traceable(name="Validate SQL")
def validate_sql_node(state: ChatState) -> dict:
    is_valid, validation_error = validate_read_query(state.get("sql", ""))
    if is_valid:
        return {
            "sql_valid": True,
            "validation_error": "",
            "error": "",
        }

    retry_count = state.get("sql_retry_count", 0)
    if retry_count < MAX_SQL_RETRIES:
        return {
            "sql_valid": False,
            "validation_error": validation_error,
            "sql_retry_count": retry_count + 1,
            "error": "",
        }

    return {
        "sql_valid": False,
        "validation_error": validation_error,
        "error": f"SQL validation failed after {MAX_SQL_RETRIES} retries: {validation_error}",
    }


def validate_sql_route(state: ChatState) -> str:
    if state.get("sql_valid"):
        return "valid"
    if state.get("sql_retry_count", 0) <= MAX_SQL_RETRIES and not state.get("error"):
        return "retry"
    return "failed"


@traceable(name="SQL Failure Response")
def sql_failure_node(state: ChatState) -> dict:
    answer = (
        "I could not generate a valid read-only SQL query for that request after "
        f"{MAX_SQL_RETRIES} retries. Last validation error: "
        f"{state.get('validation_error', 'Unknown validation error.')}"
    )
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


@traceable(name="Execute SQL")
def execute_sql_node(state: ChatState) -> dict:
    if state.get("error"):
        return {"sql_result": []}

    try:
        return {"sql_result": execute_read_query(state["sql"])}
    except Exception as exc:
        return {
            "sql_result": [],
            "error": str(exc),
        }


@traceable(name="Explain SQL Result")
def explain_result_node(state: ChatState) -> dict:
    result_payload = state.get("error") or json.dumps(
        state.get("sql_result", []),
        default=str,
        indent=2,
    )
    prompt = build_result_explanation_prompt(
        question=state["user_message"],
        sql=state.get("sql", ""),
        result=result_payload,
        chat_history=_chat_history(state.get("messages")),
    )
    answer = explain_result(prompt)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }
