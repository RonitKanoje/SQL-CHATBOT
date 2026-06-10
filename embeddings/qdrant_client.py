from functools import lru_cache
from types import SimpleNamespace

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


load_dotenv()

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_CONFIG = genai_types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")

gemini_client = genai.Client()
client = QdrantClient(url="http://localhost:6333")


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    result = gemini_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=EMBEDDING_CONFIG,
    )
    return [embedding.values for embedding in result.embeddings]


@lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    return tuple(embed_documents([text])[0])


def embed_query(text: str) -> list[float]:
    return list(_embed_query_cached(text))


def clear_embedding_cache() -> None:
    _embed_query_cached.cache_clear()


embeddings = SimpleNamespace(
    embed_documents=embed_documents,
    embed_query=embed_query,
)


@lru_cache(maxsize=8)
def create_vector_store(collection_name: str) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        validate_collection_config=False,
    )


def embedding_dim() -> int:
    return len(embed_query("dimension probe"))


if __name__ == "__main__":
    print("EMBEDDING MODEL:", GEMINI_EMBEDDING_MODEL)
