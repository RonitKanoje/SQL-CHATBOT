from functools import lru_cache

from langsmith import traceable
from embeddings.qdrant_client import create_vector_store


COLLECTION_NAME = "DATABASE" ########
VECTOR_STORE = create_vector_store(COLLECTION_NAME)


@lru_cache(maxsize=4)
def _retriever_for_k(k: int):
    return VECTOR_STORE.as_retriever(
        search_kwargs={
            "k": k,
        }
    )


@traceable(name="Retrieve Schema Embeddings")
def retrieve_embed(text: str, k: int = 3):
    return _retriever_for_k(k).invoke(text)


retrieveEmbed = retrieve_embed
