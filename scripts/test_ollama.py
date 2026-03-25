#!/usr/bin/env python3
"""
Smoke test for the Ollama integration.

Run:
    python scripts/test_ollama.py

Checks:
    1. Ollama server is reachable
    2. Required models are pulled
    3. LLM generation works
    4. Embedding generation works
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.llm_client import (
    check_ollama_health,
    get_embeddings,
    get_llm,
    list_models,
)


def _header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main() -> None:
    _header("FreightIQ — Ollama Smoke Test")

    # 1. Health check
    print("\n[1/4] Checking Ollama server connectivity …")
    if not check_ollama_health():
        print("  ❌  Cannot reach Ollama at http://localhost:11434")
        print("      → Make sure Ollama is running: 'ollama serve' or open the Ollama app")
        sys.exit(1)
    print("  ✅  Ollama server is reachable")

    # 2. List models
    print("\n[2/4] Listing available models …")
    models = list_models()
    if not models:
        print("  ⚠️   No models found. Pull a model first:")
        print("      → ollama pull llama3.1:8b")
        print("      → ollama pull nomic-embed-text")
        sys.exit(1)
    for m in models:
        print(f"  • {m}")

    # 3. LLM generation
    print("\n[3/4] Testing LLM generation …")
    try:
        llm = get_llm()
        response = llm.invoke("Say 'FreightIQ is online' and nothing else.")
        content = response.content if hasattr(response, "content") else str(response)
        print(f"  ✅  LLM response: {content.strip()[:120]}")
    except Exception as exc:
        print(f"  ❌  LLM generation failed: {exc}")
        print("      → Make sure the model is pulled: 'ollama pull llama3.1:8b'")
        sys.exit(1)

    # 4. Embedding generation
    print("\n[4/4] Testing embedding generation …")
    try:
        embeddings = get_embeddings()
        vec = embeddings.embed_query("supply chain disruption test")
        print(f"  ✅  Embedding dimensions: {len(vec)}")
    except Exception as exc:
        print(f"  ❌  Embedding generation failed: {exc}")
        print("      → Make sure the model is pulled: 'ollama pull nomic-embed-text'")
        sys.exit(1)

    _header("All checks passed ✅  — Ollama is ready for FreightIQ")


if __name__ == "__main__":
    main()
