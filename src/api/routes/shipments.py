"""
Shipment risk routes.

GET /shipments/risk     — all shipments ranked by risk score
GET /shipments/{id}     — risk detail for a single shipment
"""

from __future__ import annotations

import os
from typing import List
from fastapi import APIRouter, HTTPException
import pandas as pd

from src.api.schemas import ShipmentRiskResponse
from src.ml.scorer import RiskScorer
from src.ml.feature_engineering import FEATURE_COLS

router = APIRouter(prefix="/shipments", tags=["shipments"])


def _load_shipments() -> pd.DataFrame:
    """Load shipments from CSV."""
    path = os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@router.get("/risk", response_model=List[ShipmentRiskResponse])
def list_shipment_risks():
    """Score and return all shipments ranked by risk (highest first)."""
    df = _load_shipments()
    if df.empty:
        return []

    try:
        scorer = RiskScorer.load_from_env()
        scores = scorer.score(df)

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            score_data = scores[i]
            results.append(ShipmentRiskResponse(
                shipment_id=row.get("shipment_id", f"SHP-{i}"),
                route=row.get("route", "Unknown"),
                carrier=row.get("carrier", "Unknown"),
                region=row.get("region", "Unknown"),
                risk_score=score_data["risk_score"],
                risk_level=score_data["risk_level"],
                explanation=score_data["explanation"],
            ))

        # Sort by risk score descending
        results.sort(key=lambda r: r.risk_score, reverse=True)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@router.get("/{shipment_id}", response_model=ShipmentRiskResponse)
def get_shipment_risk(shipment_id: str):
    """Get risk detail for a single shipment by ID."""
    df = _load_shipments()
    if df.empty:
        raise HTTPException(status_code=404, detail="No shipment data available")

    match = df[df["shipment_id"] == shipment_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")

    try:
        scorer = RiskScorer.load_from_env()
        row = match.iloc[0]
        score_data = scorer.score(row.to_dict())

        return ShipmentRiskResponse(
            shipment_id=row["shipment_id"],
            route=row.get("route", "Unknown"),
            carrier=row.get("carrier", "Unknown"),
            region=row.get("region", "Unknown"),
            risk_score=score_data["risk_score"],
            risk_level=score_data["risk_level"],
            explanation=score_data["explanation"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
