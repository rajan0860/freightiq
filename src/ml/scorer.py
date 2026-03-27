from __future__ import annotations

import os
import xgboost as xgb
import shap
import pandas as pd

from src.ml.feature_engineering import prepare_inference_data

class RiskScorer:
    """
    Loads the trained XGBoost model and provides an interface to score 
    new shipments and generate SHAP explanations for the scores.
    """
    
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Initialize SHAP explainer
        # Since it's a tree model, TreeExplainer is extremely fast
        self.explainer = shap.TreeExplainer(self.model)

    @classmethod
    def load_from_env(cls) -> RiskScorer:
        """Load using the configured MODEL_PATH env var."""
        model_path = os.getenv("MODEL_PATH", "./data/models/xgboost_risk.json")
        return cls(model_path)

    def score(self, shipment_data: dict | pd.DataFrame) -> dict | list[dict]:
        """
        Score one or more shipments.
        
        Returns:
            Dictionary (if passed a dict) or list of dicts (if passed a DF) with:
            {
                "risk_score": float (0-1),
                "risk_level": "LOW", "MEDIUM", or "HIGH",
                "explanation": str (SHAP-derived human readable explanation)
            }
        """
        X = prepare_inference_data(shipment_data)
        
        # Predict probability of class 1 (delay)
        probas = self.model.predict_proba(X)[:, 1]
        
        # Compute SHAP values for the prediction
        shap_values = self.explainer.shap_values(X)
        
        results = []
        for i in range(len(X)):
            score = float(probas[i])
            
            if score >= 0.7:
                level = "HIGH"
            elif score >= 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"
                
            # Generate human-readable explanation based on top SHAP values
            explanation = self._generate_explanation(X.iloc[i], shap_values[i])
            
            results.append({
                "risk_score": round(score, 2),
                "risk_level": level,
                "explanation": explanation
            })
            
        return results[0] if isinstance(shipment_data, dict) else results
        
    def _generate_explanation(self, features: pd.Series, shap_vals: list[float]) -> str:
        """
        Convert SHAP values into a human-readable English explanation.
        """
        # Pair feature names with their SHAP impact
        impacts = list(zip(features.index, shap_vals))
        
        # Sort by absolute impact magnitude to find what drove the model the most
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 2 drivers
        top_drivers = impacts[:2]
        
        reasons = []
        for feat, val in top_drivers:
            # Format feature name nicely
            nice_feat = feat.replace('_', ' ').title()
            
            if val > 0:
                reasons.append(f"high risk contribution from {nice_feat} (+{val:.2f})")
            else:
                reasons.append(f"lowered risk from {nice_feat} ({val:.2f})")
                
        return "Explanation: " + ", ".join(reasons).capitalize()
