"""
embedder.py
-----------
Wraps the HuggingFace SentenceTransformer model so it is loaded ONCE
at startup and reused for every request — not reloaded per call.

Model: all-MiniLM-L6-v2
  - Size  : ~90 MB (downloaded on first run, cached in ~/.cache/huggingface/)
  - Output: 384-dimensional float vectors
  - Speed : ~500 sentences/sec on CPU — fast enough for this use case

Why a singleton?
  Loading a transformer model takes ~2–5 seconds. If we called
  SentenceTransformer() inside each request handler, every upload and
  every chat message would pay that penalty. Instead, we load once via
  Python module-level state and reuse the same instance.
"""

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once when the FastAPI process starts.
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model. Safe to call multiple times —
    only instantiates on the first call.
    """
    global _model
    if _model is None:
        print(f"[Embedder] Loading model '{_MODEL_NAME}'...")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"[Embedder] Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of text strings. Used during PDF ingestion.

    Args:
        texts: List of chunk strings from pdf_processor.extract_texts()

    Returns:
        List of 384-dimensional float vectors (one per input text).
        Each vector is a plain Python list[float] — Pinecone expects this format.

    Note:
        SentenceTransformer.encode() returns a numpy ndarray. We call .tolist()
        to convert to plain Python lists for JSON serialisation compatibility.
    """
    model = get_model()
    # show_progress_bar=True prints a tqdm bar during large batch uploads
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string. Used during chat to embed the user's question.

    Args:
        text: The user's question as a plain string.

    Returns:
        A single 384-dimensional float vector as list[float].

    Note:
        We call encode() with a list of one string and take index [0]
        to keep the same code path as embed_documents, ensuring the model
        sees the same tokenisation logic.
    """
    model = get_model()
    embedding = model.encode([text], convert_to_numpy=True)
    return embedding[0].tolist()
