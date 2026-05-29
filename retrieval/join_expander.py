import json
import math
from pathlib import Path

from langchain_core.documents import Document
from langsmith import traceable

from embeddings.qdrant_client import embed_documents, embed_query


METADATA_PATH = Path(__file__).resolve().parents[1] / "metadata" / "schema_metadata.json"
DEFAULT_THRESHOLD = 0.55


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def load_schema_metadata(metadata_path: Path = METADATA_PATH) -> list[dict]:
    with open(metadata_path, encoding="utf-8") as file:
        return json.load(file)


def schema_documents_by_table() -> dict[str, Document]:
    docs = {}
    for item in load_schema_metadata():
        table_name = item["metadata"]["table_name"]
        docs[table_name] = Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
    return docs


@traceable(name="Expand Connected Tables")
def expand_connected_tables(
    query: str,
    retrieved_docs: list[Document],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[Document], list[dict]]:
    docs_by_table = schema_documents_by_table()
    selected_tables = {
        doc.metadata.get("table_name")
        for doc in retrieved_docs
        if doc.metadata.get("table_name")
    }

    candidate_tables = []
    for table_name in selected_tables:
        table_doc = docs_by_table.get(table_name)
        if not table_doc:
            continue
        for connected_table in table_doc.metadata.get("connected_tables", []):
            if connected_table not in selected_tables and connected_table in docs_by_table:
                candidate_tables.append((table_name, connected_table))

    if not candidate_tables:
        return retrieved_docs, []

    query_embedding = embed_query(query)
    candidate_docs = [docs_by_table[table] for _, table in candidate_tables]
    candidate_embeddings = embed_documents([doc.page_content for doc in candidate_docs])

    expanded_docs = list(retrieved_docs)
    join_candidates = []
    seen_tables = set(selected_tables)

    for (source_table, connected_table), doc, embedding in zip(
        candidate_tables,
        candidate_docs,
        candidate_embeddings,
    ):
        score = _cosine_similarity(query_embedding, embedding)
        if score < threshold or connected_table in seen_tables:
            continue

        expanded_docs.append(doc)
        seen_tables.add(connected_table)
        join_candidates.append(
            {
                "source_table": source_table,
                "connected_table": connected_table,
                "similarity": round(score, 4),
            }
        )

    return expanded_docs, join_candidates
