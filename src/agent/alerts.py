"""
Alert formatting and output utilities for the FreightIQ agent.

Converts raw Alert objects into structured JSON and human-readable summaries.
"""

from __future__ import annotations

import json
from typing import List

from src.agent.state import Alert


def format_alerts_json(alerts: List[Alert]) -> str:
    """Serialize a list of Alert objects to pretty-printed JSON."""
    return json.dumps([a.model_dump() for a in alerts], indent=2)


def format_alerts_summary(alerts: List[Alert]) -> str:
    """Generate a human-readable summary of alerts for console / dashboard."""
    if not alerts:
        return "✅ No disruption alerts at this time."

    lines = [f"🚨 {len(alerts)} Disruption Alert(s) Detected\n{'='*50}"]

    for i, alert in enumerate(alerts, 1):
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(alert.severity, "⚪")
        lines.append(
            f"\n{icon} Alert #{i}: {alert.event}\n"
            f"   Shipment:    {alert.shipment_id}\n"
            f"   Severity:    {alert.severity}\n"
            f"   Risk Score:  {alert.risk_score:.2f}\n"
            f"   Action:      {alert.recommended_action}\n"
            f"   Confidence:  {alert.confidence:.0%}"
        )

    return "\n".join(lines)


def severity_rank(severity: str) -> int:
    """Return a numeric rank for sorting (higher = more urgent)."""
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(severity, 0)


def sort_alerts_by_severity(alerts: List[Alert]) -> List[Alert]:
    """Return alerts sorted from most to least severe."""
    return sorted(alerts, key=lambda a: severity_rank(a.severity), reverse=True)
