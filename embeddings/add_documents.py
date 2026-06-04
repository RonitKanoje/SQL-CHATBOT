import json

from langchain_core.documents import Document

from embeddings.qdrant_client import create_vector_store

COLLECTION_NAME = "DATABASE"

with open(
    "metadata/schema_metadata.json",
    "r",
    encoding="utf-8-sig"
) as f:
    data = json.load(f)

documents = [
    Document(
        page_content=item["page_content"],
        metadata=item["metadata"]
    )
    for item in data
]

vector_store = create_vector_store(
    COLLECTION_NAME
)

vector_store.add_documents(documents)

print(
    f"Inserted {len(documents)} documents"
)