from __future__ import annotations

import pandas as pd
from typing import Tuple

# The numerical features we use for XGBoost training and inference
FEATURE_COLS = [
    "carrier_reliability",
    "region_disruption_count",
    "days_to_delivery",
    "weather_severity",
    "route_risk_score",
    "cargo_value_usd",
    "news_sentiment_score",
]

TARGET_COL = "is_delayed"

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from the raw shipments dataframe.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")
        
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Fill any missing values with medians if necessary (synthetic data shouldn't have any)
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y

def prepare_inference_data(data: dict | pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a single record or multiple records for inference.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
        
    # Ensure all required columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features for inference: {missing}")
        
    return df[FEATURE_COLS]
