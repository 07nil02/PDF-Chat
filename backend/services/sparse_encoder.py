"""
sparse_encoder.py
-----------------
Generates BM25 sparse vectors for hybrid search.

BM25 (Best Match 25) is the standard keyword relevance algorithm used by
Elasticsearch, Solr, and most search engines. It scores terms by:
  - Term frequency (how often a word appears in this chunk)
  - Inverse document frequency (how rare the word is across all chunks)
  - Document length normalization

Pinecone's BM25Encoder handles all of this. It must be FIT on your corpus
(the PDF chunks) before it can encode — similar to fitting a sklearn scaler.
The fitted encoder is stored in memory as a module-level singleton.

Output format: { "indices": [token_ids...], "values": [bm25_scores...] }
This is a sparse vector — most dimensions are 0, only matching terms are nonzero.
"""

import nltk

# Ensure required NLTK resources are available
for resource in ["tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        package = resource.split("/")[-1]
        print(f"[SparseEncoder] Downloading NLTK package '{package}'...")
        nltk.download(package, quiet=True)

from pinecone_text.sparse import BM25Encoder

_encoder: BM25Encoder | None = None


def fit_encoder(texts: list[str]) -> None:
    """
    Fit the BM25 encoder on the corpus of document chunks.
    Must be called ONCE after PDF ingestion, before any encoding.

    Args:
        texts: All chunk texts from the uploaded PDF.
               The encoder learns term frequencies across this corpus.
    """
    global _encoder
    print(f"[SparseEncoder] Fitting BM25 on {len(texts)} chunks...")
    _encoder = BM25Encoder()
    _encoder.fit(texts)
    print("[SparseEncoder] BM25 encoder fitted and ready.")


def encode_documents(texts: list[str]) -> list[dict]:
    """
    Encode document chunks into sparse BM25 vectors for indexing.

    Args:
        texts: List of chunk strings to encode.

    Returns:
        List of sparse vector dicts: [{"indices": [...], "values": [...]}, ...]
    """
    if _encoder is None:
        raise RuntimeError(
            "BM25 encoder not fitted. Call fit_encoder(texts) before encoding."
        )
    return _encoder.encode_documents(texts)


def encode_query(text: str) -> dict:
    """
    Encode a single query string into a sparse BM25 vector for retrieval.

    Note: encode_queries (plural) is used for a single query too —
    Pinecone's API returns a list, so we take index [0].

    Args:
        text: The user's question string.

    Returns:
        Sparse vector dict: {"indices": [...], "values": [...]}
    """
    if _encoder is None:
        raise RuntimeError(
            "BM25 encoder not fitted. Call fit_encoder(texts) before encoding."
        )
    # encode_queries returns a list — take the first element
    return _encoder.encode_queries(text)
