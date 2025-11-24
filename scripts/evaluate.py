# scripts/evaluate.py
"""
Simple evaluation stage that reads the model and the processed file, computes metrics again,
and prints/logs them. This is intentionally tiny and demonstrative.
"""

import os
import pandas as pd
import joblib

PROCESSED = "/opt/airflow/data/processed.csv"
MODEL_PATH = "/opt/airflow/data/models/random_forest.pkl"

def main():
    if not os.path.exists(PROCESSED):
        raise FileNotFoundError(PROCESSED)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    df = pd.read_csv(PROCESSED)
    X = df.drop(columns=["target"])
    y = df["target"]

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)
    from sklearn.metrics import mean_squared_error
    import numpy as np
    mse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(mse))

    # Log to stdout (Airflow task log). If you want to log to MLflow, do that in train.py.
    print(f"[evaluate] full-data RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
