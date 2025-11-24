# scripts/train.py
"""
Train a simple sklearn model and log to MLflow.
Reads:  /opt/airflow/data/processed.csv
Writes: model to /opt/airflow/data/models/random_forest.pkl and logs artifact to MLflow.
"""

import os
import joblib
import pandas as pd

def train_main():
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np

    PROCESSED = "/opt/airflow/data/processed.csv"
    MODEL_DIR = "/opt/airflow/data/models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(PROCESSED):
        raise FileNotFoundError(f"Processed data missing: {PROCESSED}")

    df = pd.read_csv(PROCESSED)
    if "target" not in df.columns:
        raise RuntimeError("Processed file missing 'target' column")

    X = df.drop(columns=["target"])
    y = df["target"]

    # small train/test split; deterministic seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("m5_forecast_experiment")

    with mlflow.start_run():
        params = {"n_estimators": 50, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)

        artifact_path = os.path.join(MODEL_DIR, "random_forest.pkl")
        joblib.dump(model, artifact_path)

        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_artifact(artifact_path, artifact_path="models")

        print(f"[train] Model saved to {artifact_path}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    train_main()
