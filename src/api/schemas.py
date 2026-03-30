"""
Pydantic request/response models for the FreightIQ API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Body for POST /query."""
    question: str = Field(..., min_length=3, description="Natural language question")


class IngestRequest(BaseModel):
    """Body for POST /ingest (optional overrides)."""
    days_back: int = Field(default=2, ge=1, le=30)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_available: bool = False
    model_loaded: bool = False


class AlertResponse(BaseModel):
    shipment_id: str
    event: str
    severity: str
    risk_score: float
    recommended_action: str
    confidence: float


class ShipmentRiskResponse(BaseModel):
    shipment_id: str
    route: str
    carrier: str
    region: str
    risk_score: float
    risk_level: str
    explanation: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int = 0
    alerts_generated: int = 0
