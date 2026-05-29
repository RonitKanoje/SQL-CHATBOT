import os
import re
from typing import Any

import pymysql
from dotenv import load_dotenv


load_dotenv()

READ_ONLY_STARTERS = ("select", "with", "show", "describe", "desc", "explain")
BLOCKED_SQL_WORDS = re.compile(
    r"\b(insert|update|delete|create|alter|drop|truncate|replace|merge|grant|revoke|call)\b",
    re.IGNORECASE,
)


def clean_sql(sql: str) -> str:
    cleaned = sql.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned.rstrip(";")


def is_read_only_sql(sql: str) -> bool:
    cleaned = clean_sql(sql).lower()
    if not cleaned.startswith(READ_ONLY_STARTERS):
        return False
    return BLOCKED_SQL_WORDS.search(cleaned) is None


def validate_read_query(sql: str) -> tuple[bool, str]:
    cleaned_sql = clean_sql(sql)
    if not is_read_only_sql(cleaned_sql):
        return False, "Only read-only SQL queries are allowed."

    validation_sql = cleaned_sql
    if cleaned_sql.lower().startswith(("select", "with")):
        validation_sql = f"EXPLAIN {cleaned_sql}"

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(validation_sql)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.DictCursor,
    )


def execute_read_query(sql: str, limit: int = 100) -> list[dict[str, Any]]:
    cleaned_sql = clean_sql(sql)
    if not is_read_only_sql(cleaned_sql):
        raise ValueError("Only read-only SQL queries are allowed.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(cleaned_sql)
            rows = cursor.fetchmany(limit)
            return [dict(row) for row in rows]
