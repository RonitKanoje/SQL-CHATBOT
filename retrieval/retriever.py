from langsmith import traceable
from embeddings.qdrant_client import create_vector_store


COLLECTION_NAME = "DATABASE" ########


@traceable(name="Retrieve Schema Embeddings")
def retrieve_embed(text: str, k: int = 5):
    vectorstore = create_vector_store(COLLECTION_NAME)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
        }
    )

    return retriever.invoke(text)


retrieveEmbed = retrieve_embed
