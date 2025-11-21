# scripts/etl_tasks.py

import pandas as pd
from sqlalchemy import create_engine
from airflow.providers.postgres.hooks.postgres import PostgresHook # Airflow Hook is essential for connections

# --- Configuration ---
POSTGRES_CONN_ID = "postgres_default" # This ID must be set in the Airflow UI

def get_db_engine():
    """Returns a SQLAlchemy engine using Airflow's connection details."""
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    # Get connection string from hook
    conn_string = hook.get_uri().replace("postgresql://", "postgresql+psycopg2://")
    return create_engine(conn_string)

def create_staging_tables():
    """Creates a minimal schema for raw data loading."""
    engine = get_db_engine()
    
    # We only need to define a simple feature_store table structure here for the final output
    # In a real project, this would be complex DDL.
    create_features_sql = """
    CREATE TABLE IF NOT EXISTS feature_store (
        id VARCHAR PRIMARY KEY,
        date DATE,
        item_id VARCHAR,
        store_id VARCHAR,
        sell_price FLOAT,
        wm_yr_wk INTEGER,
        month INTEGER,
        day_of_week INTEGER,
        snap_ca INTEGER,
        lag_28_sales FLOAT,
        target_sales FLOAT
    );
    """
    with engine.connect() as conn:
        conn.execute(create_features_sql)
        conn.commit()
    print("Staging tables created/verified.")


def load_raw_data_and_feature_engineer(**kwargs):
    """
    Loads M5 data from disk, performs feature engineering, and loads the final table 
    back into the feature_store table using a simplified UPSERT (Idempotency).
    """
    # Assuming CSV files are mounted to the container at this path
    DATA_PATH = '/opt/airflow/data/' 
    
    # 1. Load Data
    sales_df = pd.read_csv(DATA_PATH + 'sales_train_validation.csv')
    calendar_df = pd.read_csv(DATA_PATH + 'calendar.csv')
    prices_df = pd.read_csv(DATA_PATH + 'sell_prices.csv')

    # 2. Melt Sales Data (Convert wide to long format)
    # This transforms the 30K x 1913 grid into a long time series table
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_df = sales_df.melt(
        id_vars=id_vars,
        var_name='d',
        value_name='target_sales'
    )
    
    # 3. Merge Datasets for Feature Creation
    df = sales_df.merge(calendar_df, on='d', how='left')
    df = df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    # 4. Feature Engineering (Simplified for demonstration)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Lagged Feature: Critical for time series forecasting
    # We only calculate the lag for the specific item/store combination.
    df['lag_28_sales'] = df.groupby(['id'])['target_sales'].shift(28)
    
    # Select final columns and handle NaN/Infinity
    final_df = df[[
        'id', 'date', 'item_id', 'store_id', 'sell_price', 'wm_yr_wk', 
        'month', 'day_of_week', 'snap_ca', 'lag_28_sales', 'target_sales'
    ]].fillna(0) # Simple imputation

    # 5. Load to DB (The Load part of ETL)
    engine = get_db_engine()
    # The 'if_exists="replace"' provides *simple* idempotency for demonstration 
    # (replaces the whole table daily). A production system would use 'UPSERT' logic.
    final_df.to_sql(
        'feature_store', 
        engine, 
        if_exists='replace', 
        index=False, 
        method='multi',
        chunksize=5000 # Important for large datasets
    )
    print(f"Successfully loaded {len(final_df)} rows into feature_store.")