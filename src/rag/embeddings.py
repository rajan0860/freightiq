"""
Embedding setup for the FreightIQ RAG pipeline.

Uses Ollama local embeddings (nomic-embed-text by default) via the
central LLM client.  The ``get_embedding_function`` helper returns a
LangChain-compatible embedding object ready for ChromaDB.
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from src.llm import get_embeddings as _get_embeddings


def get_embedding_function() -> OllamaEmbeddings:
    """Return the project-wide embedding function.

    This is the single entry-point that ``vector_store.py`` and any other
    module needing embeddings should call.
    """
    return _get_embeddings()
