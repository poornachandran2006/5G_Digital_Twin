"""
Train and save the Anomaly Detector on the full KPI dataset.
Run this once from the project root:
    python scripts/train_anomaly_detector.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.anomaly_detector import AnomalyDetector, FEATURE_COLUMNS

DATA_PATH = os.path.join("data", "kpi_dataset.csv")
MODEL_PATH = os.path.join("models", "anomaly_detector.pkl")


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # Keep only the 18 feature columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in dataset: {missing}")
        sys.exit(1)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    print(f"  Feature matrix shape: {X.shape}")

    # Train
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X)

    # Save
    os.makedirs("models", exist_ok=True)
    detector.save(MODEL_PATH)
    print(f"  Saved to {MODEL_PATH}")

    # Quick sanity check
    normal_row = X[100]   # pick a random row from the middle (likely normal)
    result = detector.score(normal_row)
    print(f"\nSanity check on row 100: {result}")
    print("Done. anomaly_detector.pkl is ready.")


if __name__ == "__main__":
    main()