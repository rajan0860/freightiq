#!/usr/bin/env python3
"""
Train the XGBoost risk model using the synthetic dataset.

Usage:
    python scripts/train_model.py
"""

import sys
import os
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.feature_engineering import prepare_training_data
from src.ml.train import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost risk model")
    parser.add_argument("--data", default="data/synthetic/shipments.csv", help="Path to input CSV")
    parser.add_argument("--output", default="data/models/", help="Output directory for model")
    args = parser.parse_args()

    print("Loading data...")
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file {args.data} not found. Run scripts/generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(args.data)
    X, y = prepare_training_data(df)
    
    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    print("Training XGBoost risk model...")
    
    trainer = ModelTrainer()
    metrics = trainer.train(X, y)
    
    print(f"Train AUC:  {metrics['train_auc']:.3f}")
    print(f"Val AUC:    {metrics['val_auc']:.3f}")
    print(f"Precision:  {metrics['precision']:.3f}")
    print(f"Recall:     {metrics['recall']:.3f}")
    
    model_path = trainer.save(args.output)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
