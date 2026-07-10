import json
from pathlib import Path

from langchain_core.documents import Document
from langsmith import traceable
from qdrant_client.http.models import Distance, VectorParams

from embeddings.qdrant_client import client, create_vector_store, embedding_dim


COLLECTION_NAME = "DATABASE"
SCHEMA_METADATA_PATH = Path(__file__).resolve().parents[1] / "metadata" / "schema_metadata.json"


@traceable(name="Create Static Schema Collection")
def create_collection_if_not_exists() -> None:
    if client.collection_exists(COLLECTION_NAME):
        return

    vector_size = embedding_dim()

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def load_schema_documents(metadata_path: Path = SCHEMA_METADATA_PATH) -> list[Document]:
    with open(metadata_path, encoding="utf-8-sig") as f:
        schema_metadata = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in schema_metadata
    ]


@traceable(name="Store Static Schema Embeddings")
def store_schema_embeddings() -> None:
    create_collection_if_not_exists()

    collection_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    if collection_count > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {collection_count} records.")
        return

    vectorstore = create_vector_store(COLLECTION_NAME)
    docs = load_schema_documents()
    vectorstore.add_documents(docs)
    print(f"Stored {len(docs)} schema documents in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    store_schema_embeddings()
