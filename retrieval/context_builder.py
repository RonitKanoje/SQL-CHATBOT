from langchain_core.documents import Document
from langsmith import traceable


@traceable(name="Build SQL Context")
def build_context(docs: list[Document], join_candidates: list[dict]) -> str:
    table_sections = []
    for doc in docs:
        table_name = doc.metadata.get("table_name", "unknown")
        table_sections.append(f"### {table_name}\n{doc.page_content}")

    join_lines = [
        (
            f"- {item['source_table']} -> {item['connected_table']} "
            f"(similarity: {item['similarity']})"
        )
        for item in join_candidates
    ]

    joins = "\n".join(join_lines) if join_lines else "- No extra connected tables passed threshold."

    return (
        "Use only the schema information below when writing SQL.\n\n"
        "Relevant tables:\n"
        + "\n\n".join(table_sections)
        + "\n\nConnected table expansion:\n"
        + joins
    )
