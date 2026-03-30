"""
LangGraph agent node functions for the FreightIQ disruption detection pipeline.

Three nodes:
  1. detect_node  — parse raw feeds into structured DisruptionEvent objects
  2. retrieve_node — pull relevant context from the RAG vector store
  3. recommend_node — generate actionable alerts using LLM + risk scores
"""

from __future__ import annotations

import json
import pandas as pd
from typing import Dict, Any

from src.llm.llm_client import get_llm
from src.rag.retriever import DisruptionRetriever
from src.rag.prompts import DETECT_PROMPT, RECOMMEND_PROMPT
from src.ml.scorer import RiskScorer
from src.ml.feature_engineering import FEATURE_COLS
from src.agent.state import AgentState, DisruptionEvent, ScoredShipment, Alert


# ---------------------------------------------------------------------------
# Node 1: Detect
# ---------------------------------------------------------------------------

def detect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse raw news + weather feeds and extract structured disruption events."""
    raw_feeds = state.get("raw_feeds", [])

    if not raw_feeds:
        return {"events": [], "error": None}

    # Format the raw feeds into a readable block for the LLM
    feed_text = "\n\n".join(
        f"[{item.get('type', 'unknown').upper()}] {item.get('title', '')}\n"
        f"{item.get('description', '')}"
        for item in raw_feeds
    )

    llm = get_llm(temperature=0.0)
    prompt = DETECT_PROMPT.format(raw_feeds=feed_text)

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from the LLM response
        events_raw = _parse_json_from_response(content)

        events = []
        for evt in events_raw:
            events.append(DisruptionEvent(
                event_summary=evt.get("event_summary", "Unknown event"),
                event_type=evt.get("event_type", "port_congestion"),
                region=evt.get("region", "Unknown"),
                severity=evt.get("severity", "MEDIUM"),
                estimated_delay_days=evt.get("estimated_delay_days", 0),
                affected_routes=evt.get("affected_routes", []),
            ))

        return {"events": events, "error": None}

    except Exception as e:
        return {"events": [], "error": f"Detect node failed: {e}"}


# ---------------------------------------------------------------------------
# Node 2: Retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant historical context from ChromaDB for detected events."""
    events = state.get("events", [])

    if not events:
        return {"context": "No events detected — no context to retrieve."}

    retriever = DisruptionRetriever()

    all_docs = []
    for event in events:
        query = f"{event.event_summary} {event.region} {event.event_type}"
        docs = retriever.query(query, k=3)
        all_docs.extend(docs)

    # Remove duplicates based on page_content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    context = retriever.format_docs(unique_docs) if unique_docs else "No historical context found."
    return {"context": context}


# ---------------------------------------------------------------------------
# Node 2.5: Score Shipments (runs between retrieve and recommend)
# ---------------------------------------------------------------------------

def score_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Score the current shipments using the XGBoost risk model."""
    shipments = state.get("shipments", [])

    if not shipments:
        return {"scored_shipments": []}

    try:
        scorer = RiskScorer.load_from_env()
        df = pd.DataFrame(shipments)

        # Only include records that have all required features
        valid = df.dropna(subset=FEATURE_COLS)
        if valid.empty:
            return {"scored_shipments": []}

        scores = scorer.score(valid)
        if isinstance(scores, dict):
            scores = [scores]

        scored = []
        for i, row in valid.iterrows():
            score_data = scores[i] if i < len(scores) else scores[0]
            scored.append(ScoredShipment(
                shipment_id=row.get("shipment_id", f"SHP-{i}"),
                route=row.get("route", "Unknown"),
                carrier=row.get("carrier", "Unknown"),
                risk_score=score_data["risk_score"],
                risk_level=score_data["risk_level"],
                explanation=score_data["explanation"],
            ))

        return {"scored_shipments": scored, "error": None}

    except Exception as e:
        return {"scored_shipments": [], "error": f"Score node failed: {e}"}


# ---------------------------------------------------------------------------
# Node 3: Recommend
# ---------------------------------------------------------------------------

def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate actionable alerts using the LLM given events, context, and scores."""
    events = state.get("events", [])
    context = state.get("context", "")
    scored_shipments = state.get("scored_shipments", [])

    if not events:
        return {"alerts": []}

    # Format inputs for the LLM prompt
    events_text = json.dumps(
        [e.model_dump() if hasattr(e, "model_dump") else e for e in events],
        indent=2,
    )
    shipments_text = json.dumps(
        [s.model_dump() if hasattr(s, "model_dump") else s for s in scored_shipments],
        indent=2,
    ) if scored_shipments else "No shipments scored."

    llm = get_llm(temperature=0.0)
    prompt = RECOMMEND_PROMPT.format(
        events=events_text,
        context=context,
        scored_shipments=shipments_text,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        alerts_raw = _parse_json_from_response(content)

        alerts = []
        for a in alerts_raw:
            alerts.append(Alert(
                shipment_id=a.get("shipment_id", "UNKNOWN"),
                event=a.get("event", "Unknown disruption"),
                severity=a.get("severity", "MEDIUM"),
                risk_score=float(a.get("risk_score", 0.0)),
                recommended_action=a.get("recommended_action", "Monitor situation"),
                confidence=float(a.get("confidence", 0.5)),
            ))

        return {"alerts": alerts, "error": None}

    except Exception as e:
        return {"alerts": [], "error": f"Recommend node failed: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_from_response(text: str) -> list:
    """Best-effort extraction of a JSON array from LLM output."""
    # Try direct parse first
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # Try to find JSON between ```json ... ``` markers
    import re
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []
