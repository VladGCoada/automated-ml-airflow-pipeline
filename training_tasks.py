# scripts/training_tasks.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from .etl_tasks import get_db_engine # Reuse DB connection logic

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://mlflow_server:5000" # Assumes MLflow service in Docker
MODEL_NAME = "M5SalesForecaster"

def train_and_register_model(**kwargs):
    """
    Pulls data from DB, trains XGBoost, logs to MLflow, and registers the model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("M5_Daily_Retraining")
    
    # 1. Load Data from Feature Store
    engine = get_db_engine()
    query = "SELECT * FROM feature_store WHERE target_sales IS NOT NULL AND lag_28_sales > 0"
    df = pd.read_sql(query, engine)
    
    # Focus only on a small subset for quick training during MLOps demo
    df = df.sample(frac=0.01, random_state=42) 
    
    # Prepare Data
    FEATURES = ['sell_price', 'wm_yr_wk', 'month', 'day_of_week', 'snap_ca', 'lag_28_sales']
    TARGET = 'target_sales'
    
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train XGBoost Model
    with mlflow.start_run(run_name=f"XGB_Run_{kwargs['ds']}") as run:
        
        # Hyperparameters (Simple set)
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        regressor = xgb.XGBRegressor(**params)
        regressor.fit(X_train, y_train)
        
        # 3. Evaluate and Log Metrics
        y_pred = regressor.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        print(f"Model trained. RMSE: {rmse}")

        # 4. Log and Register Model
        mlflow.sklearn.log_model(
            sk_model=regressor, 
            artifact_path="model", 
            registered_model_name=MODEL_NAME
        )
        
        # 5. Transition Model to Staging/Production (Requires MlflowClient setup in Airflow)
        # We simulate this step here:
        print(f"Model version automatically created. Now check MLflow UI to promote to Production.")

    return rmse