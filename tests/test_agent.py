"""
Tests for the LangGraph disruption agent.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.state import DisruptionEvent, ScoredShipment, Alert, AgentState
from src.agent.alerts import (
    format_alerts_summary,
    format_alerts_json,
    severity_rank,
    sort_alerts_by_severity,
)


class TestAgentState:
    """Tests for the Pydantic state schemas."""

    def test_disruption_event_creation(self):
        event = DisruptionEvent(
            event_summary="Port strike at Rotterdam",
            event_type="labour_dispute",
            region="Europe",
            severity="HIGH",
            estimated_delay_days=5,
            affected_routes=["Shanghai → Rotterdam"],
        )
        assert event.event_summary == "Port strike at Rotterdam"
        assert event.severity == "HIGH"
        assert len(event.affected_routes) == 1

    def test_alert_defaults(self):
        alert = Alert(
            shipment_id="SHP-001",
            event="Storm in Pacific",
            severity="MEDIUM",
            risk_score=0.65,
            recommended_action="Monitor weather updates",
        )
        assert alert.confidence == 0.0  # default
        assert alert.risk_score == 0.65

    def test_agent_state_defaults(self):
        state = AgentState()
        assert state.raw_feeds == []
        assert state.events == []
        assert state.alerts == []
        assert state.error is None


class TestAlertFormatting:
    """Tests for the alert formatting utilities."""

    def _sample_alerts(self):
        return [
            Alert(
                shipment_id="SHP-001",
                event="Rotterdam strike",
                severity="HIGH",
                risk_score=0.88,
                recommended_action="Reroute via Hamburg",
                confidence=0.9,
            ),
            Alert(
                shipment_id="SHP-002",
                event="Shanghai typhoon",
                severity="MEDIUM",
                risk_score=0.55,
                recommended_action="Hold shipment",
                confidence=0.7,
            ),
            Alert(
                shipment_id="SHP-003",
                event="Minor customs delay",
                severity="LOW",
                risk_score=0.2,
                recommended_action="Monitor situation",
                confidence=0.5,
            ),
        ]

    def test_format_alerts_summary_with_alerts(self):
        alerts = self._sample_alerts()
        summary = format_alerts_summary(alerts)

        assert "3 Disruption Alert(s)" in summary
        assert "Rotterdam strike" in summary
        assert "SHP-001" in summary
        assert "🔴" in summary  # HIGH icon
        assert "🟡" in summary  # MEDIUM icon
        assert "🟢" in summary  # LOW icon

    def test_format_alerts_summary_empty(self):
        summary = format_alerts_summary([])
        assert "No disruption alerts" in summary

    def test_format_alerts_json(self):
        import json
        alerts = self._sample_alerts()
        json_str = format_alerts_json(alerts)
        parsed = json.loads(json_str)
        assert len(parsed) == 3
        assert parsed[0]["shipment_id"] == "SHP-001"

    def test_severity_rank(self):
        assert severity_rank("HIGH") == 3
        assert severity_rank("MEDIUM") == 2
        assert severity_rank("LOW") == 1
        assert severity_rank("UNKNOWN") == 0

    def test_sort_alerts_by_severity(self):
        alerts = self._sample_alerts()
        # Reverse them first
        reversed_alerts = list(reversed(alerts))
        sorted_alerts = sort_alerts_by_severity(reversed_alerts)

        assert sorted_alerts[0].severity == "HIGH"
        assert sorted_alerts[1].severity == "MEDIUM"
        assert sorted_alerts[2].severity == "LOW"


class TestAgentNodes:
    """Tests for individual agent node functions."""

    def test_detect_node_empty_feeds(self):
        """detect_node should return empty events for empty feeds."""
        from src.agent.nodes import detect_node

        result = detect_node({"raw_feeds": []})
        assert result["events"] == []

    @patch("src.agent.nodes.RiskScorer")
    def test_score_node_empty_shipments(self, mock_scorer):
        """score_node should return empty list for no shipments."""
        from src.agent.nodes import score_node

        result = score_node({"shipments": []})
        assert result["scored_shipments"] == []

    def test_recommend_node_no_events(self):
        """recommend_node should return empty alerts when no events."""
        from src.agent.nodes import recommend_node

        result = recommend_node({"events": [], "context": "", "scored_shipments": []})
        assert result["alerts"] == []

    def test_parse_json_from_response(self):
        """_parse_json_from_response should extract JSON from various formats."""
        from src.agent.nodes import _parse_json_from_response

        # Direct JSON
        assert _parse_json_from_response('[{"key": "val"}]') == [{"key": "val"}]

        # Wrapped in markdown
        text = '```json\n[{"a": 1}]\n```'
        assert _parse_json_from_response(text) == [{"a": 1}]

        # Invalid
        assert _parse_json_from_response("no json here") == []
