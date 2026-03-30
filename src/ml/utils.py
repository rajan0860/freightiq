"""
Utility helpers for the FreightIQ ML pipeline.
"""

from __future__ import annotations

import os
import json
import pandas as pd
from typing import List, Optional


def load_shipments(path: Optional[str] = None) -> pd.DataFrame:
    """Load the shipments CSV from the default or given path."""
    path = path or os.getenv("SHIPMENTS_PATH", "data/synthetic/shipments.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shipments file not found: {path}")
    return pd.read_csv(path)


def load_feature_names(path: str = "data/models/features.json") -> List[str]:
    """Load the saved feature column names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def risk_level_from_score(score: float) -> str:
    """Convert a risk probability (0-1) to a categorical level."""
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    return "LOW"


def risk_color(level: str) -> str:
    """Return a hex color for a risk level (useful for dashboards)."""
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(level, "#64748b")


def summarize_risk_distribution(df: pd.DataFrame, score_col: str = "risk_score") -> dict:
    """Return a summary dict of risk distribution from scored shipments."""
    if score_col not in df.columns:
        return {}
    levels = df[score_col].apply(risk_level_from_score)
    return {
        "total": len(df),
        "high": int((levels == "HIGH").sum()),
        "medium": int((levels == "MEDIUM").sum()),
        "low": int((levels == "LOW").sum()),
        "avg_score": round(float(df[score_col].mean()), 3),
        "max_score": round(float(df[score_col].max()), 3),
    }
