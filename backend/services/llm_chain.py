"""
llm_chain.py
------------
Builds and invokes the LangChain RAG chain with support for conversational history.

Pipeline:
  PromptTemplate  →  ChatGroq (llama-3.1-8b-instant)  →  StrOutputParser
"""

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

RAG_PROMPT_TEMPLATE = """You are an expert assistant that answers questions based strictly on the provided context extracted from a PDF document.

Context from the document:
{context}

Conversation History:
{history}

Question: {question}

Instructions:
- Answer using ONLY the information in the context above.
- If the context does not contain enough information, respond with: "I don't have enough information in the provided document to answer this."
- Be concise, precise, and cite which part of the document your answer comes from when relevant.
- Do not hallucinate or add information not present in the context.
- Use markdown formatting for clarity (bullet points, bold text) when it improves readability.

Answer:"""

CONDENSE_PROMPT_TEMPLATE = """Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question, in its original language, that can be understood without the conversation history. Do NOT answer the question, just rewrite it or return it as is if it is already standalone.

Conversation History:
{history}

Follow-up Question: {question}
Standalone Question:"""


_llm: ChatGroq | None = None
_chain = None
_condense_chain = None


def _get_llm() -> ChatGroq:
    """Lazy initialize the ChatGroq model instance."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found in environment. "
                "Check your .env file."
            )
        print("[LLMChain] Initializing Groq LLM (llama-3.1-8b-instant)...")
        _llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,        # low temperature = factual, less creative
            max_tokens=1024,        # cap output length; the answer should be concise
        )
    return _llm


def get_chain():
    """
    Build and return the RAG chain. Lazily initialized on first call.

    Chain: prompt_template | llm | output_parser

    Returns:
        A LangChain Runnable that accepts {"context": str, "question": str, "history": str}
        and returns a plain string answer.
    """
    global _chain

    if _chain is not None:
        return _chain

    llm = _get_llm()

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "history"],
    )

    _chain = prompt | llm | StrOutputParser()
    print("[LLMChain] Chain ready.")
    return _chain


def get_condense_chain():
    """
    Build and return the condense question chain. Lazily initialized on first call.
    """
    global _condense_chain

    if _condense_chain is not None:
        return _condense_chain

    llm = _get_llm()

    prompt = PromptTemplate(
        template=CONDENSE_PROMPT_TEMPLATE,
        input_variables=["history", "question"],
    )

    _condense_chain = prompt | llm | StrOutputParser()
    print("[LLMChain] Condense chain ready.")
    return _condense_chain


def condense_question(history: list[dict], question: str) -> str:
    """
    Rewrite a follow-up question to be standalone using conversation history.
    If history is empty, returns original question.
    """
    if not history:
        return question

    from services.memory import format_history_for_prompt
    history_str = format_history_for_prompt(history)

    condense_chain = get_condense_chain()
    standalone = condense_chain.invoke({
        "history": history_str,
        "question": question,
    })

    return standalone.strip()


def generate_answer(
    context_chunks: list[dict],
    question: str,
    history: list[dict] = None,
) -> str:
    """
    Generate an answer given retrieved context chunks, a user question, and optional history.

    Args:
        context_chunks: List of dicts from vector_store.similarity_search()
                        Each dict has a "text" key with the chunk content.
        question      : The user's raw question string.
        history       : Optional list of previous chat messages.

    Returns:
        str — the LLM's answer as a plain string.
    """
    chain = get_chain()

    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        page_ref = f"(Page {chunk['page'] + 1})" if chunk.get("page") is not None else ""
        context_parts.append(
            f"[Chunk {i} {page_ref}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    from services.memory import format_history_for_prompt
    history_str = format_history_for_prompt(history) if history else "No previous conversation."

    # Invoke the chain — this is a synchronous call to the Groq API
    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": history_str,
    })

    return answer
