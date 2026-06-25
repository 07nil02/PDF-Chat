"""
embedder.py
-----------
Uses Google Gemini gemini-embedding-2 via the free Google AI Studio API.

MTEB score: highest quality available at $0.
Output dimensions: 768

IMPORTANT: Your Pinecone index must be recreated with dimension=768.
Delete the old index in the Pinecone dashboard and create a new one:
  Name   : pdf-chatbot
  Dims   : 768
  Metric : cosine
"""

import os
import google.generativeai as genai

_configured = False

def _configure():
    global _configured
    if not _configured:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        _configured = True
        print("[Embedder] Gemini gemini-embedding-2 ready.")


def get_model():
    """
    Eagerly configures the Gemini API.
    Maintained for compatibility with main.py startup initialization.
    """
    _configure()
    return None


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of document chunks.
    Gemini's task_type='RETRIEVAL_DOCUMENT' tells the model these are
    passages being indexed — produces better retrieval vectors than
    using the default task type.

    Batches in groups of 100 to stay within API limits.
    """
    _configure()
    all_embeddings = []
    BATCH_SIZE = 100

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=batch,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768,
        )
        all_embeddings.extend(result["embedding"])
        print(f"[Embedder] Embedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """
    Embed a single user query.
    task_type='RETRIEVAL_QUERY' is a different instruction to the model
    than RETRIEVAL_DOCUMENT — this asymmetry is what makes retrieval
    accurate. Never use RETRIEVAL_DOCUMENT for queries.
    """
    _configure()
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=text,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    )
    return result["embedding"]