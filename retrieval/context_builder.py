import re
from langchain_core.documents import Document
from langsmith import traceable


SECTION_HEADER = re.compile(r"^[A-Za-z][A-Za-z ]+:\s*$")


def _extract_bullets(page_content: str, section_name: str) -> list[str]:
    section_header = f"{section_name.lower()}:"
    in_section = False
    bullets = []

    for line in page_content.splitlines():
        stripped = line.strip()
        if not in_section:
            if stripped.lower() == section_header:
                in_section = True
            continue

        if SECTION_HEADER.match(stripped) and not stripped.startswith("-"):
            break
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())

    return bullets


def _column_names(page_content: str) -> list[str]:
    columns = []
    for item in _extract_bullets(page_content, "Columns"):
        column_name = item.split(":", 1)[0].strip()
        if column_name:
            columns.append(column_name)
    return columns


def _foreign_keys(page_content: str) -> list[str]:
    return _extract_bullets(page_content, "Relationships")


@traceable(name="Build SQL Context")
def build_context(docs: list[Document], join_candidates: list[dict]) -> str:
    table_sections = []
    for doc in docs:
        table_name = doc.metadata.get("table_name", "unknown")
        columns = _column_names(doc.page_content)
        foreign_keys = _foreign_keys(doc.page_content)
        connected_tables = doc.metadata.get("connected_tables", [])

        table_sections.append(
            "\n".join(
                [
                    f"Table Name: {table_name}",
                    f"Columns: {', '.join(columns) if columns else 'None listed'}",
                    f"Primary Key: {doc.metadata.get('primary_key', 'None listed')}",
                    "Foreign Keys:",
                    "\n".join(f"- {item}" for item in foreign_keys) if foreign_keys else "- None listed",
                    (
                        "Connected Tables: "
                        + (", ".join(connected_tables) if connected_tables else "None listed")
                    ),
                ]
            )
        )

    join_lines = [
        f"- {item['source_table']} -> {item['connected_table']}"
        for item in join_candidates
    ]

    joins = "\n".join(join_lines) if join_lines else "- No extra connected tables were added."

    return (
        "Use only the schema information below when writing SQL.\n\n"
        "Relevant Tables:\n"
        + "\n\n".join(table_sections)
        + "\n\nConnected Table Expansion:\n"
        + joins
    )
