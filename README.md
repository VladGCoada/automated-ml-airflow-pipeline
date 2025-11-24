automated_ml_airflow_pipeline/
│
├── dags/
│   ├── m5_sales_pipeline.py
│   └── utils/
│        ├── __init__.py
│        └── etl_tasks_module.py
│
├── mlruns/              
│
├── scripts/              
│   ├── training_tasks.py
│   ├── preprocessing.py
│   └── modelling.py
│
├── m5data/                
│
├── docker-compose.yaml
├── requirements.txt
├── .env
├── README.md
└── .gitignore

A[Raw M5 Dataset (CSV)] --> B[Airflow DAG: load_data]
B --> C[Preprocessing & Feature Engineering]
C --> D[Train Model (Random Forest / Prophet)]
D --> E[Evaluate Model]
E --> F[MLflow Tracking Server]

F --> G[Model Registry]
G --> H[Production Model]
G --> I[Staging / Archived Models]

subgraph Airflow
    B
    C
    D
    E
end

subgraph MLflow
    F
    G
end

F --> J[Model Artifacts (S3 / local)]

STRUCTURE :


├── dags/
│   ├── m5_sales_pipeline.py
│   └── utils/
│        ├── __init__.py
│        └── etl_tasks_module.py
│
├── mlruns/              
│
├── scripts/              
│   ├── training_tasks.py
│   ├── preprocessing.py
│   └── modelling.py
│
├── m5data/                
│
├── docker-compose.yaml
├── requirements.txt
├── .env
├── README.md
└── .gitignore
