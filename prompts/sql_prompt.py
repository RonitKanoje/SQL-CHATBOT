SQL_GENERATION_PROMPT = """You are a MySQL expert for a read-only SQL chatbot.

Rules:
- Generate exactly one MySQL SELECT query.
- Read-only queries only: SELECT, WITH, SHOW, DESCRIBE, or EXPLAIN.
- Never generate INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, TRUNCATE, REPLACE, GRANT, REVOKE, or CALL.
- Use only tables and columns present in the schema context.
- Add joins only when relationships are present in the schema context.
- Prefer clear aliases for multi-table queries.
- Return only SQL. Do not include markdown fences or explanations.

Conversation history:
{chat_history}

Schema context:
{context}

User question:
{question}

Previous failed attempt:
{validation_feedback}
"""


SQL_RESULT_EXPLANATION_PROMPT = """You explain SQL query results to a business user.

Rules:
- Answer in natural language.
- Be concise and directly answer the user question.
- Use only the query result and chat history.
- If there are no rows, say that no matching records were found.
- If there is an execution error, explain it simply without inventing data.

Conversation history:
{chat_history}

User question:
{question}

Generated SQL:
{sql}

SQL result:
{result}
"""


def build_sql_generation_prompt(
    question: str,
    context: str,
    chat_history: str = "",
    validation_feedback: str = "",
) -> str:
    return SQL_GENERATION_PROMPT.format(
        question=question,
        context=context,
        chat_history=chat_history or "No previous conversation.",
        validation_feedback=validation_feedback or "None.",
    )


def build_result_explanation_prompt(
    question: str,
    sql: str,
    result: str,
    chat_history: str = "",
) -> str:
    return SQL_RESULT_EXPLANATION_PROMPT.format(
        question=question,
        sql=sql,
        result=result,
        chat_history=chat_history or "No previous conversation.",
    )
