import logging
import pandas as pd
import numpy as np
import os

log = logging.getLogger(__name__)

def load_data_and_feature_engineer(**kwargs):
    """
    Loads raw M5 data, performs necessary feature engineering (minimal in this example),
    and pushes the feature set to a local file or XCom for the next task.
    """
    ti = kwargs['ti']
    data_path = "/opt/airflow/data/sales_train_evaluation.csv"
    
    if not os.path.exists(data_path):
        log.error(f"Data file not found at: {data_path}")
        raise FileNotFoundError(f"Required CSV not found at {data_path}")
        
    log.info(f"Loading data from {data_path}")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Simple Feature Engineering (Creating a feature to make the model training realistic)
    # The actual M5 data is too large for simple XCom, so we'll simulate the feature columns 
    # needed for training here by dropping the item/store ID columns.
    
    # Assuming the first columns (id, item_id, dept_id, etc.) are features (X) 
    # and the last column (d_1913) is the target (y).
    
    # For a simple DAG demonstration, we save the processed DF to a temporary location 
    # to be read by the next task. This prevents large data transfers via XCom.
    processed_data_path = "/tmp/m5_processed_data.csv"
    df.to_csv(processed_data_path, index=False)
    
    log.info(f"Processed feature set saved to temporary path: {processed_data_path}")
    
    # XCom push the *path* to the data, not the data itself (AVOIDS XCOM OVERLOAD)
    ti.xcom_push(key="processed_data_path", value=processed_data_path)
    
    return processed_data_path