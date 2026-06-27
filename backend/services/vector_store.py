"""
vector_store.py
---------------
Manages all interactions with the Pinecone cloud vector database.

Responsibilities:
  1. Initialize the Pinecone client and connect to the index (once at startup).
  2. Upsert (insert/update) vectors during PDF ingestion.
  3. Query the index during chat to retrieve the top-k most similar chunks.

Index configuration:
  - Metric    : cosine  
  - Dimensions: 768    
  - Pod type  : Starter 
"""

import os
import uuid
from pinecone import Pinecone, ServerlessSpec

_pinecone_client: Pinecone | None = None
_index = None


def get_index():
    """
    Lazy-initialize the Pinecone client and index connection.
    """
    global _pinecone_client, _index

    if _index is not None:
        return _index

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "pdf-chatbot")

    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY not found in environment. "
            "Check your .env file."
        )

    print(f"[VectorStore] Connecting to Pinecone index '{index_name}'...")
    _pinecone_client = Pinecone(api_key=api_key)

    # If the index doesn't exist yet, create it automatically.
    existing_indexes = [idx.name for idx in _pinecone_client.list_indexes()]
    if index_name not in existing_indexes:
        print(f"[VectorStore] Index '{index_name}' not found. Creating...")
        _pinecone_client.create_index(
            name=index_name,
            dimension=768,         
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[VectorStore] Index '{index_name}' created.")

    _index = _pinecone_client.Index(index_name)
    stats = _index.describe_index_stats()
    print(f"[VectorStore] Connected. Total vectors in index: {stats['total_vector_count']}")
    return _index


def upsert_vectors(chunks: list, embeddings: list[list[float]]) -> int:
    """
    Store text chunks and their embeddings in Pinecone.

    Args:
        chunks    : List[Document] 
        embeddings: List[list[float]] 

    Returns:
        int — number of vectors successfully upserted.

    Vector format expected by Pinecone:
        { "id": str, "values": list[float], "metadata": { "text": str, ... } }

    The "text" field in metadata is critical — it's what we retrieve during
    chat to build the context string for the LLM.
    """
    index = get_index()

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    # Build the list of (id, vector, metadata) tuples Pinecone expects.
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page", 0),
                "source": chunk.metadata.get("source", "unknown"),
            },
        })

    # Pinecone recommends batch sizes of 100. 
    BATCH_SIZE = 100
    total_upserted = 0

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        response = index.upsert(vectors=batch)
        total_upserted += response.get("upserted_count", len(batch))
        print(f"[VectorStore] Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")

    print(f"[VectorStore] Total upserted: {total_upserted} vectors")
    return total_upserted


def similarity_search(query_embedding: list[float], top_k: int = 4) -> list[dict]:
    """
    Find the top-k most similar chunks to the query embedding.

    Args:
        query_embedding: 768-dim vector from embedder.embed_query()
        top_k          : Number of results to return (default 4).

    Returns:
        List of dicts, each containing:
          { "text": str, "score": float, "page": int, "source": str }
        Sorted by similarity score descending.
    """
    index = get_index()

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for match in response.get("matches", []):
        results.append({
            "text": match["metadata"].get("text", ""),
            "score": round(match["score"], 4),
            "page": match["metadata"].get("page", 0),
            "source": match["metadata"].get("source", "unknown"),
        })

    return results


def clear_index() -> None:
    """
    Delete all vectors from the index.
    Called before a new PDF is uploaded so the chatbot doesn't mix
    context from multiple documents.

    Note: This deletes ALL vectors — suitable for single-user MVP.
    For multi-user production, you'd namespace by user/session ID instead.
    """
    index = get_index()
    stats = index.describe_index_stats()
    
    # If the index is already empty, or if the default namespace does not exist,
    if stats.get('total_vector_count', 0) == 0:
        print("[VectorStore] Index is already empty. Skipping clear.")
        return

    namespaces = stats.get('namespaces', {})
    if "" not in namespaces:
        print("[VectorStore] Default namespace does not exist. Skipping clear.")
        return

    try:
        index.delete(delete_all=True)
        print("[VectorStore] All vectors cleared from index.")
    except Exception as e:
        if "Namespace not found" in str(e) or "404" in str(e):
            print("[VectorStore] Namespace not found or already empty. Skipping clear.")
        else:
            raise e

