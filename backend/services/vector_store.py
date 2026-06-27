"""
vector_store.py
---------------
Manages TWO Pinecone indexes for hybrid search:
  1. Dense index  (pdf-chatbot)        — semantic similarity via Gemini embeddings
  2. Sparse index (pdf-chatbot-sparse) — keyword relevance via BM25

Retrieval uses Reciprocal Rank Fusion (RRF) to merge both ranked lists
into a single final ranking without needing to normalize scores.

Why two indexes instead of one sparse-dense index?
  Pinecone's single hybrid index requires dotproduct metric and has known
  score normalization issues on serverless. Two separate indexes + RRF is
  Pinecone's own recommendation for serverless deployments and is what
  production systems use.
"""

import os
import uuid
from pinecone import Pinecone, ServerlessSpec

_client: Pinecone | None = None
_dense_index = None
_sparse_index = None


def _get_client() -> Pinecone:
    global _client
    if _client is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY not set in .env")
        _client = Pinecone(api_key=api_key)
    return _client


def _ensure_index(client: Pinecone, name: str, dimension: int, metric: str):
    """Create index if it doesn't exist, then return it."""
    existing = [idx.name for idx in client.list_indexes()]
    if name not in existing:
        print(f"[VectorStore] Creating index '{name}' (dim={dimension}, metric={metric})...")
        client.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return client.Index(name)


def get_dense_index():
    global _dense_index
    if _dense_index is None:
        client = _get_client()
        name = os.getenv("PINECONE_INDEX_NAME", "pdf-chatbot")
        _dense_index = _ensure_index(client, name, dimension=768, metric="cosine")
        stats = _dense_index.describe_index_stats()
        print(f"[VectorStore] Dense index ready. Vectors: {stats['total_vector_count']}")
    return _dense_index


def get_sparse_index():
    global _sparse_index
    if _sparse_index is None:
        client = _get_client()
        name = os.getenv("PINECONE_SPARSE_INDEX_NAME", "pdf-chatbot-sparse")
        # Sparse indexes use dotproduct. Dimension=1 because sparse vectors
        # don't use the dimension field — Pinecone ignores it for sparse queries.
        _sparse_index = _ensure_index(client, name, dimension=1, metric="dotproduct")
        stats = _sparse_index.describe_index_stats()
        print(f"[VectorStore] Sparse index ready. Vectors: {stats['total_vector_count']}")
    return _sparse_index


def upsert_vectors(
    chunks: list,
    dense_embeddings: list[list[float]],
    sparse_vectors: list[dict],
) -> int:
    """
    Upsert chunks into BOTH indexes simultaneously.

    Args:
        chunks          : List[Document] from pdf_processor
        dense_embeddings: List of 768-dim Gemini vectors
        sparse_vectors  : List of BM25 sparse dicts from sparse_encoder

    Returns:
        Number of vectors upserted (same count for both indexes).
    """
    dense_idx = get_dense_index()
    sparse_idx = get_sparse_index()

    BATCH_SIZE = 100
    total = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks    = chunks[i : i + BATCH_SIZE]
        batch_dense     = dense_embeddings[i : i + BATCH_SIZE]
        batch_sparse    = sparse_vectors[i : i + BATCH_SIZE]

        # Generate one shared ID per chunk so we can correlate results later
        ids = [str(uuid.uuid4()) for _ in batch_chunks]

        # Build metadata once — shared between both indexes
        metadatas = [
            {
                "text":   chunk.page_content,
                "page":   chunk.metadata.get("page", 0),
                "source": chunk.metadata.get("source", "unknown"),
            }
            for chunk in batch_chunks
        ]

        # Dense upsert — standard format
        dense_vectors = [
            {"id": id_, "values": emb, "metadata": meta}
            for id_, emb, meta in zip(ids, batch_dense, metadatas)
        ]
        dense_idx.upsert(vectors=dense_vectors)

        # Sparse upsert — uses sparse_values field, dummy dense values=[1.0]
        # Pinecone requires at least a values field even for sparse-primary indexes
        # and it must contain at least one non-zero value.
        sparse_records = [
            {
                "id": id_,
                "values": [1.0],              # placeholder dense value (non-zero)
                "sparse_values": sv,          # the actual BM25 sparse vector
                "metadata": meta,
            }
            for id_, sv, meta in zip(ids, batch_sparse, metadatas)
        ]
        sparse_idx.upsert(vectors=sparse_records)

        total += len(batch_chunks)
        print(f"[VectorStore] Upserted batch {i // BATCH_SIZE + 1} ({len(batch_chunks)} chunks) to both indexes")

    return total


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: dict,
    top_k: int = 4,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Query both indexes and merge results with Reciprocal Rank Fusion.

    RRF formula: score(chunk) = 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_sparse)

    Chunks appearing in both result sets get additive score boosts.
    Chunks in only one set still contribute — they just score lower.

    Args:
        dense_vector : 768-dim Gemini query embedding
        sparse_vector: BM25 query sparse dict
        top_k        : Final number of chunks to return after fusion
        rrf_k        : RRF damping constant (60 is the standard default)

    Returns:
        List of top_k dicts: [{"text", "score", "page", "source"}, ...]
        Sorted by RRF score descending.
    """
    dense_idx  = get_dense_index()
    sparse_idx = get_sparse_index()

    # Fetch more candidates than top_k so RRF has enough to work with
    candidate_k = min(top_k * 3, 20)

    # Query both indexes in parallel (sequential here — simple and sufficient)
    dense_results  = dense_idx.query(
        vector=dense_vector,
        top_k=candidate_k,
        include_metadata=True,
    ).get("matches", [])

    sparse_results = sparse_idx.query(
        vector=[1.0],                  # placeholder (non-zero)
        sparse_vector=sparse_vector,   # actual query
        top_k=candidate_k,
        include_metadata=True,
    ).get("matches", [])

    # ── RRF fusion ──────────────────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    chunk_metadata: dict[str, dict] = {}

    for rank, match in enumerate(dense_results):
        id_ = match["id"]
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_metadata[id_] = match.get("metadata", {})

    for rank, match in enumerate(sparse_results):
        id_ = match["id"]
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (rrf_k + rank + 1)
        if id_ not in chunk_metadata:
            chunk_metadata[id_] = match.get("metadata", {})

    # Sort by RRF score descending, take top_k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for id_, score in ranked:
        meta = chunk_metadata.get(id_, {})
        results.append({
            "text":   meta.get("text", ""),
            "score":  round(score, 6),
            "page":   meta.get("page", 0),
            "source": meta.get("source", "unknown"),
        })

    return results


def _clear_single_index(index, name: str) -> None:
    try:
        stats = index.describe_index_stats()
        if stats.get('total_vector_count', 0) == 0:
            print(f"[VectorStore] Index '{name}' is already empty. Skipping clear.")
            return

        namespaces = stats.get('namespaces', {})
        if "" not in namespaces:
            print(f"[VectorStore] Default namespace does not exist in '{name}'. Skipping clear.")
            return

        index.delete(delete_all=True)
        print(f"[VectorStore] All vectors cleared from index '{name}'.")
    except Exception as e:
        if "Namespace not found" in str(e) or "404" in str(e):
            print(f"[VectorStore] Namespace not found or already empty in '{name}'. Skipping clear.")
        else:
            raise e


def clear_indexes() -> None:
    """
    Clear both indexes before a new PDF is uploaded.
    """
    _clear_single_index(get_dense_index(), "dense")
    _clear_single_index(get_sparse_index(), "sparse")
    print("[VectorStore] Both indexes cleared.")


# ── Legacy Compatibility Functions ──────────────────────────────────────────

def get_index():
    """Legacy compatibility function for get_dense_index."""
    return get_dense_index()


def clear_index() -> None:
    """Legacy compatibility function for clear_indexes."""
    clear_indexes()


def similarity_search(query_embedding: list[float], top_k: int = 4) -> list[dict]:
    """Legacy similarity search using dense index only."""
    dense_idx = get_dense_index()
    response = dense_idx.query(
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
