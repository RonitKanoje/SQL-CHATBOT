from qdrant_client.models import (
    Distance,
    VectorParams
)

from embeddings.qdrant_client import (
    client,
    embedding_dim,
)

client.create_collection(
    collection_name="DATABASE",
    vectors_config=VectorParams(
        size=embedding_dim(),
        distance=Distance.COSINE,
    ),
)

print("Collection created")