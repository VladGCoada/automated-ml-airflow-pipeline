"""
Trains a RandomForestRegressor on the preprocessed M5 dataset.

Reads:
    /opt/airflow/data/processed_m5.parquet

Logs:
    MLflow experiment: m5_forecast_experiment
Outputs:
    /opt/airflow/data/models/m5_random_forest.pkl
"""

import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


DATA_PATH = "/opt/airflow/data/processed_m5.parquet"
MODEL_DIR = "/opt/airflow/data/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("[M5 Training] Loading preprocessed dataset…")
    df = pd.read_parquet(DATA_PATH)

    # Drop rows with early NaNs (from lags)
    df = df.dropna(subset=["lag_28", "rolling_28"])

    # Feature set
    FEATURES = [
        "sell_price",
        "lag_1", "lag_7", "lag_28",
        "rolling_7", "rolling_28",
        "wday", "week", "month", "year",
        "is_weekend"
    ]
    TARGET = "demand"

    X = df[FEATURES]
    y = df[TARGET]

    print("[M5 Training] Splitting train/test…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("m5_forecast_experiment")

    params = {
        "n_estimators": 200,
        "max_depth": 20,
        "n_jobs": -1,
        "random_state": 42
    }

    print("[M5 Training] Starting MLflow run…")
    with mlflow.start_run():
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)

        model_path = f"{MODEL_DIR}/m5_random_forest.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_artifact(model_path)

        print(f"[M5 Training] Done. MSE={mse:.4f}")
        print(f"[M5 Training] Model saved: {model_path}")


if __name__ == "__main__":
    main()
