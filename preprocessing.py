# scripts/preprocessing.py
"""
Simple preprocessing script for the M5 CSVs.
Input (mounted):  /opt/airflow/data/sales_train_evaluation.csv
Output (mounted): /opt/airflow/data/processed.csv

This script is intentionally small and deterministic so CI can run it quickly.
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH = "C:\projects\automated_ML_AIRFLOW_PIPELINE\m5data\sales_train_evaluation.csv"
OUTPUT_PATH = "/opt/airflow/data/processed.csv"

def build_simple_features(df):
    # Example: treat the rightmost column as the target (toy example)
    # and create two trivial features so training doesn't crash.
    # Real pipeline should implement lags, rolling stats, calendar/price joins, etc.
    X = pd.DataFrame()
    # Numeric features: mean and std of all numeric columns (toy)
    numeric = df.select_dtypes(include=np.number)
    X["row_mean"] = numeric.mean(axis=1)
    X["row_std"] = numeric.std(axis=1).fillna(0.0)
    # Target: last column
    y = df.iloc[:, -1].rename("target")
    result = pd.concat([X, y], axis=1)
    return result

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input CSV: {INPUT_PATH}")
    print(f"[preprocessing] Reading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[preprocessing] Raw shape: {df.shape}")
    processed = build_simple_features(df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed.to_csv(OUTPUT_PATH, index=False)
    print(f"[preprocessing] Wrote processed data to {OUTPUT_PATH} shape={processed.shape}")

if __name__ == "__main__":
    main()
