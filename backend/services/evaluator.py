"""
evaluator.py
------------
RAGAs integration for live per-request evaluation.

Scores three reference-free metrics on every chat response:
  - Faithfulness     : is the answer grounded in retrieved context?
  - Answer Relevancy : does the answer address the question?
  - Context Precision: are the retrieved chunks relevant to the question?

"""

import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
from collections import deque
from statistics import mean

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper

# Sliding window of last 20 eval results
_score_history: deque = deque(maxlen=20)

# RAGAs metric instances — initialized lazily
_metrics = None
_evaluator_llm = None
_evaluator_embeddings = None


from langchain_core.embeddings import Embeddings

class GeminiEmbeddingsWrapper(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from services.embedder import embed_documents
        return embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        from services.embedder import embed_query
        return embed_query(text)


def _get_evaluator_embeddings():
    global _evaluator_embeddings
    if _evaluator_embeddings is None:
        _evaluator_embeddings = GeminiEmbeddingsWrapper()
        print("[Evaluator] Gemini embeddings wrapper ready.")
    return _evaluator_embeddings


def _get_evaluator_llm():
    """
    Wrap a dedicated Groq LLM in RAGAs' LangchainLLMWrapper.
    Uses llama-3.3-70b-versatile for high reliability in structured parsing.
    """
    global _evaluator_llm
    if _evaluator_llm is None:
        from langchain_groq import ChatGroq
        _evaluator_llm = LangchainLLMWrapper(
            ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.3-70b-versatile",
                temperature=0.0,
            )
        )
        print("[Evaluator] RAGAs LLM wrapper ready.")
    return _evaluator_llm


def _get_metrics():
    global _metrics
    if _metrics is None:
        llm = _get_evaluator_llm()
        _metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=_get_evaluator_embeddings()),
            LLMContextPrecisionWithoutReference(llm=llm),
        ]
        print("[Evaluator] RAGAs metrics initialized.")
    return _metrics


async def score_response(
    question: str,
    answer: str,
    context_chunks: list[dict],
) -> dict:
    """
    Score a single RAG response using RAGAs reference-free metrics.

    Runs asynchronously so it doesn't block the HTTP response — the
    /chat endpoint returns the answer immediately, and scoring happens
    in the background via asyncio.create_task().

    Args:
        question      : The user's original question
        answer        : The LLM-generated answer
        context_chunks: The retrieved chunks used to generate the answer

    Returns:
        Dict of scores: {"faithfulness": float, "answer_relevancy": float,
                         "context_precision": float}
        Returns empty dict if evaluation fails (non-critical).
    """
    try:
        # RAGAs expects retrieved_contexts as list of plain strings
        contexts = [chunk["text"] for chunk in context_chunks]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        dataset = EvaluationDataset(samples=[sample])
        metrics = _get_metrics()

        # Run evaluation — this makes LLM calls internally in a separate thread pool
        result = await asyncio.to_thread(
            evaluate,
            dataset=dataset,
            metrics=metrics,
            embeddings=_get_evaluator_embeddings(),
        )
        result_df = result.to_pandas()

        # Dynamically find columns in the result dataframe to prevent key errors
        faith_col = [c for c in result_df.columns if "faithfulness" in c][0]
        relevance_col = [c for c in result_df.columns if "answer_relevancy" in c][0]
        precision_col = [c for c in result_df.columns if "precision" in c][0]

        import math

        def clean_val(v):
            try:
                val = float(v)
                if math.isnan(val) or math.isinf(val):
                    return None
                return round(val, 3)
            except (ValueError, TypeError):
                return None

        scores = {
            "faithfulness":      clean_val(result_df[faith_col].iloc[0]),
            "answer_relevancy":  clean_val(result_df[relevance_col].iloc[0]),
            "context_precision": clean_val(result_df[precision_col].iloc[0]),
            "question":          question[:100],   # truncated for display
        }

        # Store in sliding window history
        _score_history.append(scores)
        print(f"[Evaluator] Scores — F:{scores['faithfulness']} "
              f"AR:{scores['answer_relevancy']} CP:{scores['context_precision']}")

        return scores

    except Exception as e:
        # Evaluation failure must never break the chat response
        print(f"[Evaluator] Scoring failed (non-critical): {e}")
        return {}


def get_aggregate_scores() -> dict:
    """
    Return aggregate statistics across all scored responses.
    Exposed via GET /eval/scores.
    """
    if not _score_history:
        return {"message": "No evaluations run yet. Ask some questions first."}

    history = list(_score_history)

    def safe_mean(key):
        vals = [s[key] for s in history if s.get(key) is not None]
        if not vals:
            return None
        return round(mean(vals), 3)

    return {
        "sample_count": len(history),
        "averages": {
            "faithfulness":      safe_mean("faithfulness"),
            "answer_relevancy":  safe_mean("answer_relevancy"),
            "context_precision": safe_mean("context_precision"),
        },
        "latest": history[-1],
        "history": history,
    }
