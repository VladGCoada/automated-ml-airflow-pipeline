flowchart TD

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

F --> J[Model Artifacts (Local Volume)]
automated_ml_airflow_pipeline/
│
├── dags/
│   ├── m5_sales_pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── etl_tasks_module.py
│
├── scripts/
│   ├── preprocessing.py
│   ├── modelling.py
│   └── training_tasks.py
│
├── mlruns/              # MLflow experiment store (ignored in git)
├── m5data/              # Raw M5 dataset (ignored in git)
│
├── docker-compose.yaml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
Pipeline Summary

Airflow schedules and orchestrates:

Data loading

Preprocessing

Feature engineering

Model training

Model evaluation

Model registration

MLflow tracks:

Parameters

Metrics

Models

Versions in the Model Registry

Docker Compose provides reproducible infrastructure.How to Run

Install Docker Desktop.

Clone the repo:

git clone https://github.com/vladgcoada/automated-ml-airflow-pipeline.git
cd automated-ml-airflow-pipeline


Start the stack:

docker-compose up --build


Access:

Airflow UI: http://localhost:8080

MLflow UI: http://localhost:5000

Notes

Large dataset stored under m5data/, excluded with .gitignore.

MLflow artifacts stored locally under mlruns/.

Current model: Random Forest baseline (Prophet-ready structure included).

Pipeline is modular and can easily be extended to:

XGBoost / LightGBM

Prophet forecasting

Hyperparameter tuning

Cloud storage (S3/GCS/Azure)

Status

✔ DAG operational
✔ MLflow tracking working
✔ Model registry enabled
✔ Dockerized environment
⬜ Optional: advanced models (Prophet, XGBoost)
⬜ Optional: CI/CD workflows
