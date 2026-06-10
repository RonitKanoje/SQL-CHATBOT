from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from graph.nodes import (
    blocked_node,
    context_node,
    execute_sql_node,
    explain_result_node,
    generate_sql_node,
    guardrail_node,
    guardrail_route,
    retrieval_node,
    sql_failure_node,
    validate_sql_node,
    validate_sql_route,
)
from graph.state import ChatState


def build_chatbot(checkpointer=None):
    graph = StateGraph(ChatState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("context", context_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("sql_failure", sql_failure_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("explain_result", explain_result_node)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        guardrail_route,
        {
            "blocked": "blocked",
            "allowed": "retrieve",
        },
    )
    graph.add_edge("blocked", END)
    graph.add_edge("retrieve", "context")
    graph.add_edge("context", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        validate_sql_route,
        {
            "valid": "execute_sql",
            "retry": "generate_sql",
            "failed": "sql_failure",
        },
    )
    graph.add_edge("sql_failure", END)
    graph.add_edge("execute_sql", "explain_result")
    graph.add_edge("explain_result", END)

    compiled = graph.compile(checkpointer=checkpointer or MemorySaver())
    return compiled   ### 

@traceable(name="Chat Graph", run_type="chain", tags=["langgraph"])
def run_chatbot(user_message: str, thread_id: str = "test-thread 124"):
    """Wrapper function to enable LangSmith tracing for the full graph execution"""
    chatbot = build_chatbot()
    response = chatbot.invoke(
        {"user_message": user_message},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return response


if __name__ == "__main__":
    print("Starting")
    response = run_chatbot(
        "Show the top 10 customers by revenue along with their total orders, invoices generated, payments made, support tickets raised, and average order value."
    )
    print(response["answer"])