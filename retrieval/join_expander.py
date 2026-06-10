import json
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langsmith import traceable


METADATA_PATH = Path(__file__).resolve().parents[1] / "metadata" / "schema_metadata.json"


@lru_cache(maxsize=1)
def load_schema_metadata(metadata_path: Path = METADATA_PATH) -> list[dict]:
    with open(metadata_path, encoding="utf-8-sig") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def schema_documents_by_table() -> dict[str, Document]:
    docs = {}
    for item in load_schema_metadata():
        table_name = item["metadata"]["table_name"]
        docs[table_name] = Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
    return docs


@lru_cache(maxsize=1)
def schema_relationship_graph() -> dict[str, tuple[str, ...]]:
    return {
        item["metadata"]["table_name"]: tuple(item["metadata"].get("connected_tables", []))
        for item in load_schema_metadata()
    }


RELATIONSHIP_GRAPH = schema_relationship_graph()


@traceable(name="Expand Connected Tables")
def expand_connected_tables(
    query: str,
    retrieved_docs: list[Document],
) -> tuple[list[Document], list[dict]]:
    docs_by_table = schema_documents_by_table()
    selected_tables = {
        doc.metadata.get("table_name")
        for doc in retrieved_docs
        if doc.metadata.get("table_name")
    }

    candidate_tables = []
    seen_candidates = set()
    for table_name in selected_tables:
        for connected_table in RELATIONSHIP_GRAPH.get(table_name, ()):
            if connected_table not in selected_tables and connected_table in docs_by_table:
                if connected_table in seen_candidates:
                    continue
                seen_candidates.add(connected_table)
                candidate_tables.append((table_name, connected_table))

    if not candidate_tables:
        return retrieved_docs, []
    
    expanded_docs = list(retrieved_docs)
    join_candidates = []
    seen_tables = set(selected_tables)

    for source_table, connected_table in candidate_tables:
        if connected_table in seen_tables:
            continue

        expanded_docs.append(docs_by_table[connected_table])
        seen_tables.add(connected_table)
        join_candidates.append(
            {
                "source_table": source_table,
                "connected_table": connected_table,
            }
        )

    return expanded_docs, join_candidates
