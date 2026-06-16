"""
Tests for the FreightIQ Model Context Protocol (MCP) server.
"""

from __future__ import annotations

import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.mcp_server import (
    query_supply_chain,
    get_alerts,
    get_shipments_risk,
    get_shipment_detail,
    run_pipeline,
    get_alerts_resource,
    get_shipments_resource,
)


class TestMCPServerTools:
    """Tests for individual MCP tool handlers with mocked backend logic."""

    @patch("src.mcp_server.get_llm")
    @patch("src.mcp_server.DisruptionRetriever")
    def test_query_supply_chain(self, mock_retriever_cls, mock_get_llm):
        """Test query_supply_chain tool returns the LLM response and source references."""
        # Setup mocks
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Storm reported in Atlantic."
        mock_doc.metadata = {"source": "OpenWeather", "date": "2026-06-16", "type": "weather_alert"}
        mock_retriever.query.return_value = [mock_doc]
        mock_retriever.format_docs.return_value = "[WEATHER | OpenWeather | 2026-06-16]\nStorm reported in Atlantic."
        mock_retriever_cls.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "There is a storm in the Atlantic."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Run tool
        result = query_supply_chain("Is there a storm?")
        
        assert "There is a storm in the Atlantic" in result
        assert "Sources:** OpenWeather" in result
        mock_retriever.query.assert_called_once_with("Is there a storm?", k=5)
        mock_llm.invoke.assert_called_once()

    @patch("src.mcp_server.run_ingestion_pipeline")
    @patch("src.mcp_server.DisruptionAgent")
    @patch("src.mcp_server.NewsFetcher")
    @patch("src.mcp_server.WeatherFetcher")
    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_get_alerts_run_pipeline(self, mock_exists, mock_read_csv, mock_weather, mock_news, mock_agent_cls, mock_run_ingest):
        """Test get_alerts with run_pipeline=True generates fresh alerts."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame([{"shipment_id": "SHP-123", "route": "A to B", "carrier": "DHL", "region": "EU"}])
        mock_run_ingest.return_value = None
        
        mock_news.return_value.fetch_disruption_news.return_value = [{"title": "Strike"}]
        mock_weather.return_value.fetch_weather_alerts.return_value = [{"title": "Storm"}]
        
        mock_alert = MagicMock()
        mock_alert.shipment_id = "SHP-123"
        mock_alert.event = "Strike event"
        mock_alert.severity = "HIGH"
        mock_alert.risk_score = 0.9
        mock_alert.confidence = 0.8
        mock_alert.recommended_action = "Reroute"
        
        mock_agent = MagicMock()
        mock_agent.run.return_value = [mock_alert]
        mock_agent_cls.return_value = mock_agent

        # Run tool
        result = get_alerts(run_pipeline=True)

        assert "FreightIQ Disruption Alerts" in result
        assert "SHP-123" in result
        assert "Strike event" in result
        assert "Reroute" in result
        mock_run_ingest.assert_called_once()
        mock_agent.run.assert_called_once()

    @patch("src.mcp_server.requests.get")
    def test_get_alerts_no_pipeline_api_active(self, mock_get):
        """Test get_alerts falls back to active API endpoint if pipeline=False."""
        # Setup mock API response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "shipment_id": "SHP-999",
            "event": "Backend Strike",
            "severity": "MEDIUM",
            "risk_score": 0.65,
            "confidence": 0.7,
            "recommended_action": "Wait it out"
        }]
        mock_get.return_value = mock_resp

        # Run tool
        result = get_alerts(run_pipeline=False)

        assert "from API" in result
        assert "SHP-999" in result
        assert "Backend Strike" in result
        assert "Wait it out" in result

    @patch("src.mcp_server.RiskScorer")
    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_get_shipments_risk(self, mock_exists, mock_read_csv, mock_scorer_cls):
        """Test get_shipments_risk scores and displays shipments ranked by risk score."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame([
            {"shipment_id": "SHP-100", "route": "Rotterdam to Hamburg", "carrier": "FedEx", "region": "EU"},
            {"shipment_id": "SHP-200", "route": "Miami to Houston", "carrier": "UPS", "region": "US"}
        ])

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = [
            {"risk_score": 0.35, "risk_level": "LOW", "explanation": "Low risk"},
            {"risk_score": 0.85, "risk_level": "HIGH", "explanation": "High risk"}
        ]
        mock_scorer_cls.load_from_env.return_value = mock_scorer

        # Run tool
        result = get_shipments_risk(limit=5)

        # Check ranking (SHP-200 should be listed first due to higher risk score 0.85 vs 0.35)
        assert "Miami to Houston" in result
        assert "Miami to Houston" in result.split("\n")[3]  # High risk shipment ranked first
        assert "SHP-200" in result
        assert "SHP-100" in result
        assert "0.85" in result
        assert "0.35" in result

    @patch("src.mcp_server.RiskScorer")
    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_get_shipment_detail_found(self, mock_exists, mock_read_csv, mock_scorer_cls):
        """Test get_shipment_detail returns details for a valid shipment."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame([
            {
                "shipment_id": "SHP-1042",
                "route": "Rotterdam to Hamburg",
                "carrier": "FedEx",
                "region": "EU",
                "carrier_reliability": 0.8,
                "weather_severity": 0.1,
                "route_risk_score": 0.3,
                "cargo_value_usd": 250000,
                "days_to_delivery": 3,
                "news_sentiment_score": -0.1
            }
        ])

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {"risk_score": 0.45, "risk_level": "MEDIUM", "explanation": "Medium driver"}
        mock_scorer_cls.load_from_env.return_value = mock_scorer

        # Run tool
        result = get_shipment_detail("SHP-1042")

        assert "Shipment Details: SHP-1042" in result
        assert "Rotterdam to Hamburg" in result
        assert "FedEx" in result
        assert "0.45" in result
        assert "MEDIUM" in result
        assert "Medium driver" in result

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_get_shipment_detail_not_found(self, mock_exists, mock_read_csv):
        """Test get_shipment_detail returns an error message for unknown shipment."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame([{"shipment_id": "SHP-100"}])

        result = get_shipment_detail("SHP-999")
        assert "not found" in result

    @patch("src.mcp_server.run_ingestion_pipeline")
    def test_run_pipeline(self, mock_run_ingest):
        """Test run_pipeline calls underlying ingestion pipeline."""
        result = run_pipeline()
        assert "Successfully ran" in result
        mock_run_ingest.assert_called_once()


class TestMCPServerResources:
    """Tests for raw MCP resources."""

    @patch("src.mcp_server.requests.get")
    def test_get_alerts_resource_api_active(self, mock_get):
        """Test get_alerts_resource fetches alerts from API and parses json."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"shipment_id": "SHP-123", "event": "Active Alert"}]
        mock_get.return_value = mock_resp

        result = get_alerts_resource()
        parsed = json.loads(result)
        
        assert len(parsed) == 1
        assert parsed[0]["shipment_id"] == "SHP-123"
        assert parsed[0]["event"] == "Active Alert"

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_get_shipments_resource(self, mock_exists, mock_read_csv):
        """Test get_shipments_resource constructs summaries and cargo value average."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame([
            {"shipment_id": "SHP-1", "carrier": "DHL", "region": "EU", "cargo_value_usd": 100000},
            {"shipment_id": "SHP-2", "carrier": "DHL", "region": "US", "cargo_value_usd": 200000},
            {"shipment_id": "SHP-3", "carrier": "UPS", "region": "EU", "cargo_value_usd": 300000},
        ])

        result = get_shipments_resource()
        
        assert "- **Total Shipments:** 3" in result
        assert "- **Average Cargo Value (USD):** $200,000.00" in result
        assert "DHL: 2 shipments" in result
        assert "UPS: 1 shipments" in result
        assert "EU: 2 shipments" in result
        assert "US: 1 shipments" in result
