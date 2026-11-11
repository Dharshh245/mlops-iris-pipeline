"""Q2 (c) - Log the trained model to MLflow, register it, and save an MLflow
model package (build/mlflow_model) that CI publishes as a build artefact.

Run after `dvc repro`:   python src/package_mlflow.py
View the UI:             mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import os
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MODEL_NAME = "iris-classifier"
PACKAGE_DIR = Path("build/mlflow_model")


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("iris-classification")

    params = yaml.safe_load(open("params.yaml"))
    metrics = json.loads(Path("metrics.json").read_text())
    model = joblib.load("models/model.joblib")
    sample = pd.read_csv("data/prepared/test.csv")[FEATURES].head(5)
    signature = infer_signature(sample, model.predict(sample))
    sha = os.getenv("GITHUB_SHA", "local")

    with mlflow.start_run(run_name=f"build-{sha[:7]}") as run:
        mlflow.set_tags({"git_sha": sha, "source": "dvc-pipeline"})
        mlflow.log_params({f"prepare.{k}": v for k, v in params["prepare"].items()})
        mlflow.log_params({f"train.{k}": v for k, v in params["train"].items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("plots/confusion_matrix.png")
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=sample,
            registered_model_name=MODEL_NAME,
        )
        print(f"MLflow run {run.info.run_id} logged; registered as '{MODEL_NAME}'")

    shutil.rmtree(PACKAGE_DIR, ignore_errors=True)
    mlflow.sklearn.save_model(model, PACKAGE_DIR, signature=signature, input_example=sample)
    print(f"MLflow model package written to {PACKAGE_DIR}/")


if __name__ == "__main__":
    main()
