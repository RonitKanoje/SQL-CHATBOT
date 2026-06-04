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


def embed_query(text: str) -> list[float]:
    return embed_documents([text])[0]


embeddings = SimpleNamespace(
    embed_documents=embed_documents,
    embed_query=embed_query,
)


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
