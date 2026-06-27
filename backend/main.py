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
from services.memory import get_history, add_turn, clear_history
from services.llm_chain import get_chain, generate_answer, condense_question
import asyncio
from services.evaluator import score_response, get_aggregate_scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup before the server accepts requests."""
    print("[Startup] Warming up services...")
    get_model()     
    get_dense_index()
    get_sparse_index()  
    get_chain()     
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
    question:   str
    session_id: str = "default"   # frontend sends a UUID; fallback for testing

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]     # [{ "text": str, "score": float, "page": int }]

class UploadResponse(BaseModel):
    message: str
    chunks: int

class HealthResponse(BaseModel):
    status: str

class EvalScoresResponse(BaseModel):
    sample_count: int | None = None
    averages: dict | None = None
    latest: dict | None = None
    history: list | None = None
    message: str | None = None

class EvalRequest(BaseModel):
    question: str
    answer: str
    context_chunks: list[dict]


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Liveness endpoint. Render pings this to confirm the server is alive.
    Returns 200 OK with {"status": "ok"}.
    """
    return {"status": "ok"}


@app.get("/eval/scores", response_model=EvalScoresResponse)
def get_eval_scores():
    """
    Return aggregate RAGAs scores across all chat responses this session.
    Used by the frontend dashboard and for resume metrics.
    """
    return get_aggregate_scores()


@app.post("/eval/evaluate")
async def evaluate_turn(req: EvalRequest):
    """
    Manually run RAGAs evaluation on a specific turn.
    """
    scores = await score_response(
        question=req.question,
        answer=req.answer,
        context_chunks=req.context_chunks,
    )
    if not scores:
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed. This could be due to API rate limits or formatting issues."
        )
    return scores


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = "default",   # add as a query param
):
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
        clear_history(session_id)                  # NEW — reset conversation on new PDF

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
    question   = request.question.strip()
    session_id = request.session_id

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Step 1: Get history for this session
        history = get_history(session_id)

        # Step 2: Condense question for retrieval
        # If first turn, returns question unchanged (no LLM call)
        standalone_question = condense_question(history, question)

        # Step 3: Embed the CONDENSED question for retrieval
        print(f"[Chat] Embedding query (dense): '{standalone_question[:80]}...' " if len(standalone_question) > 80 else f"[Chat] Embedding query (dense): '{standalone_question}'")
        dense_vec  = embed_query(standalone_question)

        print(f"[Chat] Encoding query (sparse): '{standalone_question[:80]}...' " if len(standalone_question) > 80 else f"[Chat] Encoding query (sparse): '{standalone_question}'")
        sparse_vec = encode_query(standalone_question)

        # Step 4: Hybrid search with the condensed question
        print("[Chat] Searching Pinecone via hybrid search...")
        context_chunks = hybrid_search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=4,
        )

        if not context_chunks:
            return {
                "answer": "No document has been uploaded yet. Please upload a PDF first.",
                "sources": [],
            }

        print(f"[Chat] Retrieved {len(context_chunks)} chunks (top score: {context_chunks[0]['score']})")

        # Step 5: Generate answer with ORIGINAL question + history
        print("[Chat] Generating answer via Groq...")
        answer = generate_answer(
            context_chunks=context_chunks,
            question=question,         # original, not condensed
            history=history,
        )

        # Step 6: Save this turn to memory
        add_turn(session_id, question, answer)

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
