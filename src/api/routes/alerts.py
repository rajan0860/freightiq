"""
Alert API routes.

GET /alerts        — list all recent alerts
GET /alerts/{id}   — get a single alert by index
"""

from __future__ import annotations

from typing import List
from fastapi import APIRouter, HTTPException

from src.api.schemas import AlertResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])

# In-memory alert store (populated by the /ingest endpoint or agent runs)
_alert_store: List[dict] = []


def set_alerts(alerts: List[dict]):
    """Replace the in-memory alert store (called after agent runs)."""
    global _alert_store
    _alert_store = alerts


def get_alert_store() -> List[dict]:
    return _alert_store


@router.get("", response_model=List[AlertResponse])
def list_alerts():
    """Return all current disruption alerts."""
    return _alert_store


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int):
    """Return a single alert by its index."""
    if alert_id < 0 or alert_id >= len(_alert_store):
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return _alert_store[alert_id]
