"""
LLM client for FreightIQ — Ollama local models via LangChain.

This module is the single source of truth for all LLM and embedding
configuration.  Every other module should import from here rather than
instantiating models directly.

Usage:
    from src.llm import get_llm, get_embeddings

    llm = get_llm()                      # ChatOllama (llama3.1:8b)
    embeddings = get_embeddings()         # OllamaEmbeddings (nomic-embed-text)
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults (overridable via .env)
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOllama:
    """Return a LangChain ChatOllama instance.

    Parameters
    ----------
    model : str, optional
        Ollama model tag.  Defaults to ``OLLAMA_MODEL`` env var or
        ``llama3.1:8b``.
    temperature : float
        Sampling temperature (0 = deterministic).
    **kwargs
        Forwarded to ``ChatOllama``.
    """
    return ChatOllama(
        base_url=_base_url(),
        model=model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        **kwargs,
    )


def get_embeddings(model: Optional[str] = None) -> OllamaEmbeddings:
    """Return a LangChain OllamaEmbeddings instance.

    Parameters
    ----------
    model : str, optional
        Embedding model tag.  Defaults to ``OLLAMA_EMBED_MODEL`` env var or
        ``nomic-embed-text``.
    """
    return OllamaEmbeddings(
        base_url=_base_url(),
        model=model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )


def check_ollama_health() -> bool:
    """Return ``True`` if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{_base_url()}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def list_models() -> list[str]:
    """Return a list of model tags available on the local Ollama server."""
    try:
        resp = requests.get(f"{_base_url()}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except (requests.ConnectionError, requests.HTTPError, KeyError):
        return []
