import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# IMPORTANT: You must ensure your MLflow server tracking URI is accessible.
# The tracking URI is typically set as an environment variable in Airflow.
import mlflow
import mlflow.xgboost

log = logging.getLogger(__name__)

def train_and_register_model(**kwargs):
    """
    Trains an XGBoost model, logs metrics/parameters to MLflow, 
    and registers the final model in the MLflow Model Registry.
    """
    log.info("Starting model training and MLflow registration...")

    # --- 1. Define MLflow Experiment ---
    # MLflow automatically handles the run context here
    with mlflow.start_run(run_name=f"XGBoost_Run_{kwargs['ti'].run_id}") as run:
        # Define the experiment name (or create it if it doesn't exist)
        experiment_name = "SalesForecastingPipeline"
        mlflow.set_experiment(experiment_name)

        # --- 2. Data Simulation (Replace with actual DB/Feature Store connection) ---
        log.info("Simulating data retrieval from Feature Store (PostgreSQL)...")
        # Creating dummy data for demonstration
        N = 1000
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.rand(N) * 10,
            'feature_2': np.random.rand(N) * 5,
        })
        # Simple linear relationship + noise
        y = 5 * X['feature_1'] + 2 * X['feature_2'] + np.random.normal(0, 5, N)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log.info(f"Data split. Training on {len(X_train)} samples.")

        # --- 3. Model Training ---
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        log.info("Model training complete.")

        # --- 4. Evaluation and Metric Logging ---
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)
        
        log.info(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # --- 5. Model Registration (The MLOps Step) ---
        
        # Define the model name in the MLflow Model Registry
        model_name = "SalesForecastXGBoostModel"
        
        # Log the model artifact and register it.
        # This function automatically saves the model and creates a new version in the registry.
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        
        log.info(f"Model logged and registered as '{model_name}'.")

        # Push the model's run_id to XCom for the next task (like deployment)
        run_id = run.info.run_id
        kwargs['ti'].xcom_push(key='mlflow_run_id', value=run_id)
        log.info(f"MLflow Run ID {run_id} pushed to XCom.")

    return "Model training, logging, and registration succeeded."