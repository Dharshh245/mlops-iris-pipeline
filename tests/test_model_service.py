from pathlib import Path

import pytest

from api.model_service import ModelService

MODEL = Path("models/model.joblib")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run `dvc repro` first")

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}


def test_predicts_known_flowers():
    svc = ModelService(str(MODEL))
    svc.load()
    out = svc.predict([SETOSA, VIRGINICA])
    assert [o["species"] for o in out] == ["setosa", "virginica"]
    assert abs(sum(out[0]["probabilities"].values()) - 1) < 1e-3


def test_not_ready_before_load():
    svc = ModelService(str(MODEL))
    assert not svc.ready
    with pytest.raises(RuntimeError):
        svc.predict([SETOSA])
