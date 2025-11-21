from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


# --- TASK FUNCTIONS -------------------------------------------------------

def load_data(**context):
    data_path = "/opt/airflow/data/sales_train_evaluation.csv"
    df = pd.read_csv(data_path)
    context['ti'].xcom_push(key="df_shape", value=df.shape)


def train_model(**context):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("m5_forecast_experiment")

    data_path = "/opt/airflow/data/sales_train_evaluation.csv"
    df = pd.read_csv(data_path)

    # VERY simple example model
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X, y)

        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "random_forest_model")

        # save locally inside the container
        model_output_path = "/opt/airflow/data/models/random_forest.pkl"
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)


def validate_environment():
    print("Python and MLflow versions:")
    import sys
    import mlflow
    print(sys.version)
    print("MLflow:", mlflow.__version__)


# --- DAG DEFINITION -------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_training_pipeline",
    description="M5 forecasting model pipeline with MLflow",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
) as dag:

    env_check = PythonOperator(
        task_id="validate_environment",
        python_callable=validate_environment,
    )

    load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    echo_path = BashOperator(
        task_id="debug_paths",
        bash_command="echo Data folder: /opt/airflow/data && ls -la /opt/airflow/data",
    )

    env_check >> load >> train >> echo_path
