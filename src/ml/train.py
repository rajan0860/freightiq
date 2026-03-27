from __future__ import annotations

import os
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pandas as pd

from src.ml.feature_engineering import FEATURE_COLS

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train the XGBoost model and return evaluation metrics."""
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Predict
        y_pred_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            "train_auc": roc_auc_score(y_train, y_pred_proba_train),
            "val_auc": roc_auc_score(y_val, y_pred_proba_val),
            "precision": precision_score(y_val, y_pred_val, zero_division=0),
            "recall": recall_score(y_val, y_pred_val, zero_division=0)
        }
        
        return metrics

    def save(self, output_dir: str):
        """Save the trained model to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "xgboost_risk.json")
        self.model.save_model(model_path)
        
        # Save feature names so the scorer can validate later
        with open(os.path.join(output_dir, "features.json"), "w") as f:
            json.dump(FEATURE_COLS, f)
            
        return model_path
