"""
Agent state schema for the LangGraph disruption detection agent.

Defines the typed state that flows between the Detect → Retrieve → Recommend
nodes in the agent graph.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DisruptionEvent(BaseModel):
    """A single detected disruption event."""
    event_summary: str
    event_type: str = Field(
        description="One of: labour_dispute, weather_event, port_congestion, "
                    "geopolitical_issue, customs_delay"
    )
    region: str
    severity: str = Field(description="LOW, MEDIUM, or HIGH")
    estimated_delay_days: int = 0
    affected_routes: List[str] = Field(default_factory=list)


class ScoredShipment(BaseModel):
    """A shipment that has been scored by the XGBoost model."""
    shipment_id: str
    route: str
    carrier: str
    risk_score: float
    risk_level: str
    explanation: str


class Alert(BaseModel):
    """A structured alert produced by the Recommend node."""
    shipment_id: str
    event: str
    severity: str
    risk_score: float
    recommended_action: str
    confidence: float = 0.0


class AgentState(BaseModel):
    """
    The mutable state object that travels through the LangGraph nodes.

    Each node reads from and writes to this state:
      - detect_node  → populates `events`
      - retrieve_node → populates `context`
      - recommend_node → populates `alerts`
    """
    # Inputs
    raw_feeds: List[dict] = Field(
        default_factory=list,
        description="Raw news + weather items from the ingestion pipeline",
    )
    shipments: List[dict] = Field(
        default_factory=list,
        description="Current shipment records to score",
    )

    # Populated by detect node
    events: List[DisruptionEvent] = Field(default_factory=list)

    # Populated by retrieve node
    context: str = ""

    # Populated by scorer (between retrieve and recommend)
    scored_shipments: List[ScoredShipment] = Field(default_factory=list)

    # Populated by recommend node
    alerts: List[Alert] = Field(default_factory=list)

    # Metadata
    error: Optional[str] = None
