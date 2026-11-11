"""Stage 2 - train: fit the classifier defined in params.yaml and save it."""
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
MODEL_PATH = Path("models/model.joblib")


def build_model(p: dict):
    if p["model"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            random_state=p["random_state"],
        )
    if p["model"] == "logistic_regression":
        return LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=p["random_state"])
    raise ValueError(f"Unknown model: {p['model']}")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["train"]
    train = pd.read_csv("data/prepared/train.csv")

    model = build_model(params)
    model.fit(train[FEATURES], train[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained {params['model']} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
