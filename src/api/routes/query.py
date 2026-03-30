"""
Natural language query route.

POST /query — ask a question against the RAG knowledge base
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import QueryRequest, QueryResponse
from src.rag.retriever import DisruptionRetriever
from src.rag.prompts import QUERY_PROMPT
from src.llm.llm_client import get_llm

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_rag(body: QueryRequest):
    """Answer a natural language question using the RAG pipeline."""
    retriever = DisruptionRetriever()
    docs = retriever.query(body.question, k=5)

    context = retriever.format_docs(docs) if docs else "No relevant context found."
    sources = list({d.metadata.get("source", "Unknown") for d in docs})

    llm = get_llm(temperature=0.0)
    prompt = QUERY_PROMPT.format(context=context, question=body.question)

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=sources,
    )
