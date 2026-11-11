"""Model loading and prediction logic, kept free of web-framework code so it is easy to test."""
import os
from pathlib import Path

import joblib
import pandas as pd

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class ModelService:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "models/model.joblib"))
        self.version = os.getenv("MODEL_VERSION", "dev")
        self.model = None

    def load(self) -> None:
        self.model = joblib.load(self.model_path)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, rows: list[dict]) -> list[dict]:
        if not self.ready:
            raise RuntimeError("Model not loaded")
        X = pd.DataFrame(rows, columns=FEATURES)
        labels = self.model.predict(X)
        probs = self.model.predict_proba(X)
        classes = [str(c) for c in self.model.classes_]
        return [
            {
                "species": str(label),
                "probabilities": {c: round(float(p), 4) for c, p in zip(classes, prob)},
            }
            for label, prob in zip(labels, probs)
        ]
