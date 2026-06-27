"""
main.py
-------
FastAPI application entry point.

Endpoints:
  GET  /health    — liveness check
  POST /upload    — ingest a PDF: extract → chunk → embed → upsert to Pinecone
  POST /chat      — answer a question: embed query → retrieve → generate
"""

import os
import shutil
import tempfile
import warnings
from contextlib import asynccontextmanager

# Suppress deprecation warnings triggered by third-party packages (like pypdf/cryptography)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env BEFORE importing services — the services read env vars at init time.
load_dotenv()

from services.pdf_processor import load_and_split, extract_texts
from services.embedder import embed_documents, embed_query, get_model
from services.sparse_encoder import fit_encoder, encode_documents, encode_query   # NEW
from services.vector_store import get_dense_index, get_sparse_index, upsert_vectors, hybrid_search, clear_indexes  # UPDATED
from services.llm_chain import get_chain, generate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup before the server accepts requests."""
    print("[Startup] Warming up services...")
    get_model()     # downloads/loads the HuggingFace model into memory
    get_dense_index()
    get_sparse_index()    # NEW — warm up sparse index connection
    get_chain()     # initializes the Groq LLM + LangChain chain
    print("[Startup] All services ready. Server accepting requests.")
    yield
    print("[Shutdown] Server shutting down.")


app = FastAPI(
    title="RAG PDF Chatbot API",
    description="Upload a PDF and ask questions about it using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)



_origins = [
    "http://localhost:5173",        
    "http://localhost:3000",        
]

frontend_url = os.getenv("FRONTEND_URL", "").strip()
if frontend_url:
    _origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]     # [{ "text": str, "score": float, "page": int }]

class UploadResponse(BaseModel):
    message: str
    chunks: int

class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Liveness endpoint. Render pings this to confirm the server is alive.
    Returns 200 OK with {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF document into the vector store.

    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Save to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        print(f"[Upload] Saved '{file.filename}' to temp path: {tmp_path}")

        # Extract and chunk
        print("[Upload] Loading and splitting PDF...")
        chunks = load_and_split(tmp_path)
        texts = extract_texts(chunks)
        print(f"[Upload] Extracted {len(chunks)} chunks from {file.filename}")

        # Dense embeddings (Gemini)
        print("[Upload] Embedding chunks...")
        dense_embeddings = embed_documents(texts)
        print(f"[Upload] Generated {len(dense_embeddings)} dense embeddings")

        # Sparse BM25 vectors — fit on THIS document's corpus, then encode
        print("[Upload] Fitting BM25 encoder on document chunks...")
        fit_encoder(texts)                          # NEW
        sparse_vectors = encode_documents(texts)    # NEW
        print(f"[Upload] Generated {len(sparse_vectors)} sparse vectors")

        # Clear previous document's vectors, then upsert new ones.
        print("[Upload] Clearing previous vectors...")
        clear_indexes()                             # was clear_index()

        print("[Upload] Upserting to Pinecone...")
        count = upsert_vectors(chunks, dense_embeddings, sparse_vectors)

        return {
            "message": f"Successfully processed '{file.filename}'",
            "chunks": count,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        print(f"[Upload] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process the PDF: {str(e)}"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"[Upload] Temp file deleted: {tmp_path}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question about the uploaded PDF.

    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question is too long. Please keep it under 1000 characters."
        )

    try:
        # Embed the question
        print(f"[Chat] Embedding query (dense): '{question[:80]}...' " if len(question) > 80 else f"[Chat] Embedding query (dense): '{question}'")
        query_dense = embed_query(question)

        # Sparse BM25 vector for query
        print(f"[Chat] Encoding query (sparse): '{question[:80]}...' " if len(question) > 80 else f"[Chat] Encoding query (sparse): '{question}'")
        query_sparse = encode_query(question)       # NEW

        # Retrieve top 4 relevant chunks via hybrid search
        print("[Chat] Searching Pinecone via hybrid search...")
        context_chunks = hybrid_search(             # was similarity_search()
            dense_vector=query_dense,
            sparse_vector=query_sparse,
            top_k=4,
        )

        if not context_chunks:
            return {
                "answer": "No document has been uploaded yet, or the index is empty. Please upload a PDF first.",
                "sources": [],
            }

        print(f"[Chat] Retrieved {len(context_chunks)} chunks (top score: {context_chunks[0]['score']})")

        # Generate answer via Groq
        print("[Chat] Generating answer via Groq...")
        answer = generate_answer(context_chunks, question)

        print(f"[Chat] Answer generated ({len(answer)} chars)")

        return {
            "answer": answer,
            "sources": context_chunks,
        }

    except Exception as e:
        print(f"[Chat] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate an answer: {str(e)}"
        )
