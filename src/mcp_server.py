"""
Model Context Protocol (MCP) server for FreightIQ.

Exposes FreightIQ supply chain capabilities (RAG query, shipment scoring,
and disruption alerts) as MCP tools and resources.
"""

from __future__ import annotations

import sys
import os
import pandas as pd
import requests
from typing import List, Optional
from mcp.server.fastmcp import FastMCP

# Ensure the project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.retriever import DisruptionRetriever
from src.rag.prompts import QUERY_PROMPT
from src.llm.llm_client import get_llm
from src.ml.scorer import RiskScorer
from src.ingestion.pipeline import run_ingestion_pipeline
from src.agent.graph import DisruptionAgent
from src.ingestion.news_fetcher import NewsFetcher
from src.ingestion.weather_fetcher import WeatherFetcher

# Initialize the FastMCP server
mcp = FastMCP("FreightIQ")


@mcp.tool()
def query_supply_chain(question: str) -> str:
    """Answer a natural language question about supply chain disruptions using the local RAG pipeline and LLM.
    
    Args:
        question: The natural language question (e.g. 'What storms are currently affecting routes?').
    """
    try:
        retriever = DisruptionRetriever()
        docs = retriever.query(question, k=5)
        
        context = retriever.format_docs(docs) if docs else "No relevant context found."
        sources = list({d.metadata.get("source", "Unknown") for d in docs})
        
        llm = get_llm(temperature=0.0)
        prompt = QUERY_PROMPT.format(context=context, question=question)
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        
        sources_str = ", ".join(sources) if sources else "None"
        return f"**Answer:** {answer}\n\n**Sources:** {sources_str}"
    except Exception as e:
        return f"Error querying RAG system: {str(e)}"


@mcp.tool()
def get_alerts(run_pipeline: bool = False) -> str:
    """Get recent supply chain disruption alerts for shipments.
    
    Args:
        run_pipeline: If True, triggers a fresh ingestion of news/weather feeds
                      and re-runs the agent risk assessments before returning alerts.
                      If False, returns existing alerts from the API backend or executes a fast local run.
    """
    try:
        if run_pipeline:
            # Run ingestion
            run_ingestion_pipeline()
            
            # Gather feeds
            raw_feeds = NewsFetcher().fetch_disruption_news(days_back=2) + WeatherFetcher().fetch_weather_alerts()
            
            # Load shipments
            shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
            if not os.path.exists(shipments_path):
                return "Error: No shipment data found. Run python scripts/generate_data.py first."
            
            df = pd.read_csv(shipments_path)
            shipments = df.head(50).to_dict("records")
                
            # Run agent
            agent = DisruptionAgent()
            alerts = agent.run(raw_feeds=raw_feeds, shipments=shipments)
            
            if not alerts:
                return "No alerts generated after running the pipeline."
                
            return _format_alerts_to_markdown(alerts)
            
        else:
            # Try to fetch from FastAPI API first if it's running
            api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
            try:
                resp = requests.get(f"{api_base}/alerts", timeout=2)
                if resp.status_code == 200:
                    alerts = resp.json()
                    if not alerts:
                        return "No active alerts in the API backend. You can set run_pipeline=True to generate alerts."
                    return _format_alerts_json_to_markdown(alerts)
            except requests.RequestException:
                pass  # Fall back to running local agent directly
                
            # Load shipments
            shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
            if not os.path.exists(shipments_path):
                return "Error: No shipment data found. Run python scripts/generate_data.py first."
                
            df = pd.read_csv(shipments_path)
            shipments = df.head(50).to_dict("records")
            
            # Run agent with local/cached news and weather
            try:
                raw_feeds = NewsFetcher().fetch_disruption_news(days_back=2) + WeatherFetcher().fetch_weather_alerts()
            except Exception:
                raw_feeds = []
                
            agent = DisruptionAgent()
            alerts = agent.run(raw_feeds=raw_feeds, shipments=shipments)
            
            if not alerts:
                return "No active alerts found. You can set run_pipeline=True to fetch fresh feeds."
                
            return _format_alerts_to_markdown(alerts)
            
    except Exception as e:
        return f"Error retrieving alerts: {str(e)}"


@mcp.tool()
def get_shipments_risk(limit: int = 20) -> str:
    """Score and return current shipments ranked by risk score (highest risk first).
    
    Args:
        limit: The maximum number of shipments to return.
    """
    try:
        shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
        if not os.path.exists(shipments_path):
            return "Error: No shipment data found. Run python scripts/generate_data.py first."
            
        df = pd.read_csv(shipments_path)
        if df.empty:
            return "No shipments in database."
            
        scorer = RiskScorer.load_from_env()
        scores = scorer.score(df)
        
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            score_data = scores[i]
            results.append({
                "shipment_id": row.get("shipment_id", f"SHP-{i}"),
                "route": row.get("route", "Unknown"),
                "carrier": row.get("carrier", "Unknown"),
                "region": row.get("region", "Unknown"),
                "risk_score": score_data["risk_score"],
                "risk_level": score_data["risk_level"],
                "explanation": score_data["explanation"],
            })
            
        results.sort(key=lambda r: r["risk_score"], reverse=True)
        results = results[:limit]
        
        # Format as markdown table
        md = [f"### 🚢 Shipment Risk Scores (Top {len(results)})"]
        md.append("| Shipment ID | Route | Carrier | Region | Risk Score | Risk Level | Explanation |")
        md.append("|---|---|---|---|---|---|---|")
        for r in results:
            md.append(f"| {r['shipment_id']} | {r['route']} | {r['carrier']} | {r['region']} | {r['risk_score']:.2f} | {r['risk_level']} | {r['explanation']} |")
            
        return "\n".join(md)
    except Exception as e:
        return f"Error retrieving shipment risks: {str(e)}"


@mcp.tool()
def get_shipment_detail(shipment_id: str) -> str:
    """Get detailed risk information and SHAP explanation for a specific shipment.
    
    Args:
        shipment_id: The unique shipment ID (e.g. 'SHP-1042').
    """
    try:
        shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
        if not os.path.exists(shipments_path):
            return "Error: No shipment data found. Run python scripts/generate_data.py first."
            
        df = pd.read_csv(shipments_path)
        match = df[df["shipment_id"] == shipment_id]
        if match.empty:
            return f"Shipment '{shipment_id}' not found."
            
        row = match.iloc[0]
        scorer = RiskScorer.load_from_env()
        score_data = scorer.score(row.to_dict())
        
        md = [f"### 📦 Shipment Details: {shipment_id}\n"]
        md.append(f"- **Route:** {row.get('route', 'Unknown')}")
        md.append(f"- **Carrier:** {row.get('carrier', 'Unknown')}")
        md.append(f"- **Region:** {row.get('region', 'Unknown')}")
        md.append(f"- **Carrier Reliability:** {row.get('carrier_reliability', 0.0):.2%}")
        md.append(f"- **Weather Severity:** {row.get('weather_severity', 0.0):.2%}")
        md.append(f"- **Route Risk Score:** {row.get('route_risk_score', 0.0):.2%}")
        md.append(f"- **Cargo Value (USD):** ${row.get('cargo_value_usd', 0.0):,.2f}")
        md.append(f"- **Days to Delivery:** {row.get('days_to_delivery', 0)}")
        md.append(f"- **News Sentiment Score:** {row.get('news_sentiment_score', 0.0):.2f}\n")
        md.append("#### Risk Assessment:")
        md.append(f"- **Risk Score:** `{score_data['risk_score']:.2f}`")
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(score_data["risk_level"], "⚪")
        md.append(f"- **Risk Level:** {icon} **{score_data['risk_level']}**")
        md.append(f"- **ML Driver Explanation:** {score_data['explanation']}")
        
        return "\n".join(md)
    except Exception as e:
        return f"Error retrieving shipment detail for {shipment_id}: {str(e)}"


@mcp.tool()
def run_pipeline() -> str:
    """Trigger the news/weather data ingestion pipeline and update the ChromaDB vector database."""
    try:
        run_ingestion_pipeline()
        return "Successfully ran ingestion pipeline and updated ChromaDB."
    except Exception as e:
        return f"Error running ingestion pipeline: {str(e)}"


@mcp.resource("freightiq://alerts")
def get_alerts_resource() -> str:
    """Return the raw JSON list of current alerts."""
    import json
    try:
        # Check API
        api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
        try:
            resp = requests.get(f"{api_base}/alerts", timeout=2)
            if resp.status_code == 200:
                return json.dumps(resp.json(), indent=2)
        except requests.RequestException:
            pass
            
        # Try local agent
        shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
        if not os.path.exists(shipments_path):
            return json.dumps({"error": "No shipment data found. Run generator first."}, indent=2)
            
        df = pd.read_csv(shipments_path)
        shipments = df.head(50).to_dict("records")
        try:
            raw_feeds = NewsFetcher().fetch_disruption_news(days_back=2) + WeatherFetcher().fetch_weather_alerts()
        except Exception:
            raw_feeds = []
            
        agent = DisruptionAgent()
        alerts = agent.run(raw_feeds=raw_feeds, shipments=shipments)
        
        alert_dicts = [a.model_dump() if hasattr(a, "model_dump") else a for a in alerts]
        return json.dumps(alert_dicts, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve alerts: {str(e)}"}, indent=2)


@mcp.resource("freightiq://shipments")
def get_shipments_resource() -> str:
    """Return an overview and stats of the shipments CSV file."""
    try:
        shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
        if not os.path.exists(shipments_path):
            return "No shipment CSV file found at " + shipments_path
            
        df = pd.read_csv(shipments_path)
        total_shipments = len(df)
        carriers = df["carrier"].value_counts().to_dict()
        regions = df["region"].value_counts().to_dict()
        avg_value = df["cargo_value_usd"].mean()
        
        stats = [
            "### FreightIQ Shipment Database Overview",
            f"- **Total Shipments:** {total_shipments}",
            f"- **Average Cargo Value (USD):** ${avg_value:,.2f}",
            "- **Active Carriers:**",
        ]
        for c, count in carriers.items():
            stats.append(f"  - {c}: {count} shipments")
        stats.append("- **Regions covered:**")
        for r, count in regions.items():
            stats.append(f"  - {r}: {count} shipments")
            
        return "\n".join(stats)
    except Exception as e:
        return f"Error loading shipments summary: {str(e)}"


@mcp.prompt()
def summarize_disruptions() -> str:
    """A prompt template to guide the LLM in summarizing shipments disruption alerts and composing client warnings."""
    return (
        "You are the FreightIQ logistics assistant. Please perform the following steps:\n"
        "1. Fetch all active alerts using the 'get_alerts' tool.\n"
        "2. Detail the top 3 highest risk shipments by score, using 'get_shipment_detail' for each to identify their specific risk drivers.\n"
        "3. Write a summary of these disruptions for our logistics manager.\n"
        "4. Compose a professional, proactive email draft that the logistics manager can send to the client for the most high-risk shipment, explaining the delay and suggesting the recommended action."
    )


def _format_alerts_to_markdown(alerts) -> str:
    md = ["### 🚨 FreightIQ Disruption Alerts\n"]
    for i, a in enumerate(alerts):
        severity = getattr(a, "severity", "MEDIUM")
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
        md.append(f"#### {icon} {getattr(a, 'event', 'Disruption')} — Shipment: {getattr(a, 'shipment_id', 'N/A')}")
        md.append(f"- **Severity:** {severity}")
        md.append(f"- **Risk Score:** {getattr(a, 'risk_score', 0.0):.2f}")
        md.append(f"- **Confidence:** {getattr(a, 'confidence', 0.0):.0%}")
        md.append(f"- **Recommended Action:** {getattr(a, 'recommended_action', 'N/A')}\n")
    return "\n".join(md)


def _format_alerts_json_to_markdown(alerts: List[dict]) -> str:
    md = ["### 🚨 FreightIQ Disruption Alerts (from API)\n"]
    for i, a in enumerate(alerts):
        severity = a.get("severity", "MEDIUM")
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
        md.append(f"#### {icon} {a.get('event', 'Disruption')} — Shipment: {a.get('shipment_id', 'N/A')}")
        md.append(f"- **Severity:** {severity}")
        md.append(f"- **Risk Score:** {a.get('risk_score', 0.0):.2f}")
        md.append(f"- **Confidence:** {a.get('confidence', 0.0):.0%}")
        md.append(f"- **Recommended Action:** {a.get('recommended_action', 'N/A')}\n")
    return "\n".join(md)


if __name__ == "__main__":
    mcp.run()
