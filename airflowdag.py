# dags/m5_sales_pipeline.py

from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import sys
import os

# Add the 'scripts' directory to the path so the tasks can be imported
sys.path.append(os.path.join(os.environ['AIRFLOW_HOME'], 'scripts'))

# Import the Python functions from the scripts
from etl_tasks import create_staging_tables, load_raw_data_and_feature_engineer
from training_tasks import train_and_register_model

# --- DAG Definition ---

@dag(
    dag_id="automated_m5_sales_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=timedelta(days=1), # Run once a day
    catchup=False,
    tags=['mlops', 'sales_forecasting', 'xgboost'],
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'email_on_failure': False,
        'email_on_retry': False,
    }
)
def sales_forecasting_pipeline():
    
    # 1. Setup Task: Ensure the necessary tables exist
    create_tables_task = PythonOperator(
        task_id='create_db_tables',
        python_callable=create_staging_tables,
    )

    # 2. ETL Task: Load data, engineer features, and save to feature_store table
    # We pass the execution date ('ds') to the Python function using templates (**kwargs)
    etl_task = PythonOperator(
        task_id='etl_feature_engineering_load',
        python_callable=load_raw_data_and_feature_engineer,
    )
    
    # 3. Training Task: Train the model using data from the DB
    # The result (RMSE) is pushed to XCom for downstream use (e.g., the report)
    train_task = PythonOperator(
        task_id='train_and_register_xgboost',
        python_callable=train_and_register_model,
    )
    
    # 4. Reporting Task: Check metrics and send notification
    # We use XCom to pull the RMSE value computed in the train_task
    report_task = EmailOperator(
        task_id='send_completion_report',
        to='your_email@example.com', # Replace with your email
        subject='Airflow M5 Sales Pipeline Report: {{ ti.xcom_pull(task_ids="train_and_register_xgboost") }}',
        html_content="""
            <h3>M5 Sales Forecasting Retraining Completed</h3>
            <p>The automated daily retraining pipeline finished successfully.</p>
            <p>New Model RMSE: <b>{{ ti.xcom_pull(task_ids="train_and_register_xgboost") }}</b></p>
            <p>Check MLflow for details on the new model version.</p>
        """,
        # This task will fail if the Email connection is not configured, but it
        # demonstrates the final reporting step.
    )

    # --- Set Task Dependencies ---
    # The DAG flow: Tables -> ETL -> Train -> Report
    create_tables_task >> etl_task >> train_task >> report_task

# Instantiate the DAG
sales_forecasting_pipeline()