

This repo contains:
- Airflow DAGs to preprocess M5 data, train a model, and evaluate it.
- Dockerfile to build a custom Airflow image with ML dependencies.
- MLflow server (local) for experiment tracking and model registry.
- GitHub Actions workflow for lint + build + smoke test.

---

## Quick links
- Airflow UI: `http://localhost:8080`
- MLflow UI: `http://localhost:5000`

---

  A[Developer] -->|push| GH[GitHub]
  GH -->|PR / CI| CI[GitHub Actions]
  CI -->|build image| DockerHub[(Container Registry)] 
  Developer -->|run| DC[Docker Compose]
  DC --> Airflow[Airflow (webserver + scheduler)]
  Airflow --> DAG[ml_training_pipeline DAG]
  DAG --> Preprocess[preprocess_m5.py]
  DAG --> Train[train_m5_model.py]
  Preprocess -->|reads/writes| Data[/opt/airflow/data (m5data)]
  Train -->|logs| MLflow[MLflow Tracking Server]
  MLflow --> Artifacts[(Artifact Store: mlruns or MinIO/S3)]
  MLflow --> Registry[Model Registry]
  Registry --> Serve[Serving / Deployment]
