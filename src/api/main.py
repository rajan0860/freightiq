"""
FastAPI application entry point for FreightIQ.

Start with:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, IngestRequest, IngestResponse
from src.api.routes.alerts import router as alerts_router, set_alerts
from src.api.routes.shipments import router as shipments_router
from src.api.routes.query import router as query_router
from src.llm.llm_client import check_ollama_health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    print("🚀 FreightIQ API starting...")
    yield
    print("👋 FreightIQ API shutting down.")


app = FastAPI(
    title="FreightIQ API",
    description="AI-powered supply chain disruption detection, "
                "shipment risk scoring, and automated alerts.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(alerts_router)
app.include_router(shipments_router)
app.include_router(query_router)


# ---------------------------------------------------------------------------
# Root-level endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """System health check."""
    model_path = os.getenv("MODEL_PATH", "./data/models/xgboost_risk.json")
    return HealthResponse(
        status="ok",
        ollama_available=check_ollama_health(),
        model_loaded=os.path.exists(model_path),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["system"])
def trigger_ingest(body: IngestRequest | None = None):
    """Manually trigger the ingestion pipeline + agent run."""
    from src.ingestion.news_fetcher import NewsFetcher
    from src.ingestion.weather_fetcher import WeatherFetcher
    from src.ingestion.pipeline import run_ingestion_pipeline
    from src.agent.graph import DisruptionAgent

    # Run ingestion
    run_ingestion_pipeline()

    # Gather feeds
    raw_feeds = (
        NewsFetcher().fetch_disruption_news(days_back=body.days_back if body else 2)
        + WeatherFetcher().fetch_weather_alerts()
    )

    # Load shipments
    shipments_path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
    shipments = []
    if os.path.exists(shipments_path):
        df = pd.read_csv(shipments_path)
        shipments = df.head(50).to_dict("records")

    # Run agent
    agent = DisruptionAgent()
    alerts = agent.run(raw_feeds=raw_feeds, shipments=shipments)

    # Store alerts in memory for the GET /alerts endpoint
    alert_dicts = [a.model_dump() if hasattr(a, "model_dump") else a for a in alerts]
    set_alerts(alert_dicts)

    return IngestResponse(
        status="success",
        documents_ingested=len(raw_feeds),
        alerts_generated=len(alerts),
    )
