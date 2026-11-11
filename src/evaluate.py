"""Stage 3 - evaluate: compute accuracy / F1 and write metrics + plots."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    model = joblib.load("models/model.joblib")
    test = pd.read_csv("data/prepared/test.csv")

    y_true = test[TARGET]
    y_pred = model.predict(test[FEATURES])

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average=params["average"])), 4),
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred))

    plots = Path("plots")
    plots.mkdir(exist_ok=True)
    # Raw predictions for `dvc plots` (confusion matrix template)
    pd.DataFrame({"actual": y_true.values, "predicted": y_pred}).to_csv(
        plots / "predictions.csv", index=False
    )
    # PNG for the CML report
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(f"Confusion matrix (acc={metrics['accuracy']}, f1={metrics['f1_score']})")
    plt.tight_layout()
    plt.savefig(plots / "confusion_matrix.png", dpi=110)


if __name__ == "__main__":
    main()
