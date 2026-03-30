#!/usr/bin/env python3
"""
Manual trigger for the FreightIQ ingestion + agent pipeline.

Usage:
    python scripts/ingest_and_run.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.news_fetcher import NewsFetcher
from src.ingestion.weather_fetcher import WeatherFetcher
from src.agent.graph import DisruptionAgent


def main():
    # Step 1: Run ingestion pipeline (upserts to ChromaDB)
    print("=" * 60)
    print("STEP 1: Running ingestion pipeline")
    print("=" * 60)
    run_ingestion_pipeline()

    # Step 2: Gather raw feeds for the agent (same sources)
    print("\n" + "=" * 60)
    print("STEP 2: Gathering live feeds for agent")
    print("=" * 60)
    news_fetcher = NewsFetcher()
    weather_fetcher = WeatherFetcher()

    raw_feeds = news_fetcher.fetch_disruption_news() + weather_fetcher.fetch_weather_alerts()
    print(f"Collected {len(raw_feeds)} feed items.")

    # Step 3: Load shipment data for scoring
    print("\n" + "=" * 60)
    print("STEP 3: Loading shipment data")
    print("=" * 60)
    shipments_path = "data/synthetic/shipments.csv"
    if os.path.exists(shipments_path):
        df = pd.read_csv(shipments_path)
        # Take a sample of shipments to score (top 20)
        shipments = df.head(20).to_dict("records")
        print(f"Loaded {len(shipments)} shipments for scoring.")
    else:
        print(f"⚠️  No shipment data at {shipments_path} — run scripts/generate_data.py first.")
        shipments = []

    # Step 4: Run the LangGraph agent
    print("\n" + "=" * 60)
    print("STEP 4: Running LangGraph disruption agent")
    print("=" * 60)
    agent = DisruptionAgent()
    alerts = agent.run_and_print(raw_feeds=raw_feeds, shipments=shipments)

    print(f"\n✅ Agent run complete. Generated {len(alerts)} alert(s).")


if __name__ == "__main__":
    main()
