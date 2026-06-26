"""
embedder.py
-----------
Uses Google Gemini gemini-embedding-2 via the official new google-genai SDK.

"""

import os
from google import genai
from google.genai import types

_client: genai.Client | None = None

def _configure():
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env")
        _client = genai.Client(api_key=api_key)
        print("[Embedder] Google GenAI SDK initialized with gemini-embedding-2.")


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
        
        # Workaround: Wrap each text in a Content object to ensure
        # gemini-embedding-2 processes them as separate inputs (a batch)
        # rather than parts of a single document.
        wrapped_contents = [types.Content(parts=[types.Part(text=s)]) for s in batch]
        
        response = _client.models.embed_content(
            model="models/gemini-embedding-2",
            contents=wrapped_contents,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            )
        )
        batch_embeddings = [e.values for e in response.embeddings]
        all_embeddings.extend(batch_embeddings)
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
    response = _client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        )
    )
    return response.embeddings[0].values