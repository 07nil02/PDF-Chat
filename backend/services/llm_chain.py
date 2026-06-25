"""
llm_chain.py
------------
Builds and invokes the LangChain RAG chain.

Pipeline:
  PromptTemplate  →  ChatGroq (llama3-8b-8192)  →  StrOutputParser

Why this chain structure?
  LangChain's pipe operator (|) creates a Runnable sequence. Each stage
  receives the output of the previous stage. StrOutputParser strips the
  AIMessage wrapper and returns a plain string — easier to send as JSON.

Groq API:
  - Free tier: generous rate limits (~30 req/min, 14,400 req/day)
  - llama3-8b-8192: fast (200+ tokens/sec), 8192-token context window
  - No card required for free tier
"""

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# The prompt template — the most important tuning knob in the whole system.
# Changing this changes the personality and accuracy of every answer.
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are an expert assistant that answers questions based strictly on the provided context extracted from a PDF document.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information, respond with: "I don't have enough information in the provided document to answer this."
- Be concise, precise, and cite which part of the document your answer comes from when relevant.
- Do not hallucinate or add information not present in the context.
- Use markdown formatting for clarity (bullet points, bold text) when it improves readability.

Answer:"""


# ---------------------------------------------------------------------------
# Singleton LLM instance — loaded once, reused per request.
# ---------------------------------------------------------------------------
_llm: ChatGroq | None = None
_chain = None


def get_chain():
    """
    Build and return the RAG chain. Lazily initialized on first call.

    Chain: prompt_template | llm | output_parser

    Returns:
        A LangChain Runnable that accepts {"context": str, "question": str}
        and returns a plain string answer.
    """
    global _llm, _chain

    if _chain is not None:
        return _chain

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment. "
            "Check your .env file."
        )

    print("[LLMChain] Initializing Groq LLM (llama3-8b-8192)...")
    _llm = ChatGroq(
        api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.1,        # low temperature = factual, less creative
        max_tokens=1024,        # cap output length; the answer should be concise
    )

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    _chain = prompt | _llm | StrOutputParser()
    print("[LLMChain] Chain ready.")
    return _chain


def generate_answer(context_chunks: list[dict], question: str) -> str:
    """
    Generate an answer given retrieved context chunks and a user question.

    Args:
        context_chunks: List of dicts from vector_store.similarity_search()
                        Each dict has a "text" key with the chunk content.
        question      : The user's raw question string.

    Returns:
        str — the LLM's answer as a plain string.

    How context is assembled:
        We concatenate the text of each chunk, separated by a divider line.
        The chunk number and page reference help the LLM cite its sources.
    """
    chain = get_chain()

    # Build the context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        page_ref = f"(Page {chunk['page'] + 1})" if chunk.get("page") is not None else ""
        context_parts.append(
            f"[Chunk {i} {page_ref}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Invoke the chain — this is a synchronous call to the Groq API
    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return answer
