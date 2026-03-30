"""
Tests for the XGBoost risk scorer and feature engineering.
"""

import pytest
import os
import pandas as pd
import numpy as np


class TestFeatureEngineering:
    """Tests for feature_engineering.py functions."""

    def test_prepare_training_data_extracts_correct_columns(self):
        """prepare_training_data should return X with feature cols and y with target."""
        from src.ml.feature_engineering import prepare_training_data, FEATURE_COLS, TARGET_COL

        df = pd.DataFrame({
            "shipment_id": ["SHP-001"],
            "carrier_reliability": [0.85],
            "region_disruption_count": [2],
            "days_to_delivery": [5],
            "weather_severity": [0.3],
            "route_risk_score": [0.4],
            "cargo_value_usd": [50000],
            "news_sentiment_score": [0.1],
            "is_delayed": [1],
            "extra_col": ["ignored"],
        })

        X, y = prepare_training_data(df)

        assert list(X.columns) == FEATURE_COLS
        assert len(X) == 1
        assert y.iloc[0] == 1
        assert "extra_col" not in X.columns

    def test_prepare_training_data_raises_on_missing_target(self):
        """Should raise ValueError if target column is missing."""
        from src.ml.feature_engineering import prepare_training_data

        df = pd.DataFrame({"carrier_reliability": [0.9]})

        with pytest.raises(ValueError, match="is_delayed"):
            prepare_training_data(df)

    def test_prepare_inference_data_from_dict(self):
        """prepare_inference_data should accept a dict and return a DataFrame."""
        from src.ml.feature_engineering import prepare_inference_data, FEATURE_COLS

        data = {
            "carrier_reliability": 0.8,
            "region_disruption_count": 1,
            "days_to_delivery": 10,
            "weather_severity": 0.5,
            "route_risk_score": 0.6,
            "cargo_value_usd": 100000,
            "news_sentiment_score": -0.2,
        }

        result = prepare_inference_data(data)
        assert list(result.columns) == FEATURE_COLS
        assert len(result) == 1

    def test_prepare_inference_data_raises_on_missing_features(self):
        """Should raise ValueError if required features are missing."""
        from src.ml.feature_engineering import prepare_inference_data

        with pytest.raises(ValueError, match="Missing required features"):
            prepare_inference_data({"carrier_reliability": 0.8})


class TestRiskScorer:
    """Tests for the RiskScorer class."""

    def test_scorer_loads_model(self):
        """RiskScorer should load the saved model if it exists."""
        from src.ml.scorer import RiskScorer

        model_path = "data/models/xgboost_risk.json"
        if not os.path.exists(model_path):
            pytest.skip("Model file not found — run train_model.py first")

        scorer = RiskScorer(model_path)
        assert scorer.model is not None

    def test_scorer_returns_valid_result(self):
        """score() should return a dict with risk_score, risk_level, explanation."""
        from src.ml.scorer import RiskScorer

        model_path = "data/models/xgboost_risk.json"
        if not os.path.exists(model_path):
            pytest.skip("Model file not found — run train_model.py first")

        scorer = RiskScorer(model_path)
        result = scorer.score({
            "carrier_reliability": 0.7,
            "region_disruption_count": 3,
            "days_to_delivery": 5,
            "weather_severity": 0.8,
            "route_risk_score": 0.6,
            "cargo_value_usd": 200000,
            "news_sentiment_score": -0.5,
        })

        assert "risk_score" in result
        assert "risk_level" in result
        assert "explanation" in result
        assert 0 <= result["risk_score"] <= 1
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_scorer_handles_batch(self):
        """score() should handle a DataFrame of multiple shipments."""
        from src.ml.scorer import RiskScorer

        model_path = "data/models/xgboost_risk.json"
        if not os.path.exists(model_path):
            pytest.skip("Model file not found — run train_model.py first")

        scorer = RiskScorer(model_path)
        df = pd.DataFrame([
            {
                "carrier_reliability": 0.9,
                "region_disruption_count": 0,
                "days_to_delivery": 20,
                "weather_severity": 0.1,
                "route_risk_score": 0.2,
                "cargo_value_usd": 10000,
                "news_sentiment_score": 0.8,
            },
            {
                "carrier_reliability": 0.5,
                "region_disruption_count": 4,
                "days_to_delivery": 2,
                "weather_severity": 0.9,
                "route_risk_score": 0.8,
                "cargo_value_usd": 400000,
                "news_sentiment_score": -0.9,
            },
        ])

        results = scorer.score(df)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_scorer_file_not_found(self):
        """RiskScorer should raise FileNotFoundError for bad path."""
        from src.ml.scorer import RiskScorer

        with pytest.raises(FileNotFoundError):
            RiskScorer("/nonexistent/path/model.json")


class TestMLUtils:
    """Tests for ml/utils.py helper functions."""

    def test_risk_level_from_score(self):
        from src.ml.utils import risk_level_from_score

        assert risk_level_from_score(0.85) == "HIGH"
        assert risk_level_from_score(0.5) == "MEDIUM"
        assert risk_level_from_score(0.2) == "LOW"
        assert risk_level_from_score(0.7) == "HIGH"
        assert risk_level_from_score(0.4) == "MEDIUM"
        assert risk_level_from_score(0.39) == "LOW"

    def test_risk_color(self):
        from src.ml.utils import risk_color

        assert risk_color("HIGH") == "#ef4444"
        assert risk_color("LOW") == "#22c55e"
        assert risk_color("UNKNOWN") == "#64748b"
