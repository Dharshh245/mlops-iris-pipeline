from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not Path("models/model.joblib").exists(), reason="run `dvc repro` first"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan -> loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client):
    payload = {"instances": [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    ]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["predictions"][0]["species"] == "setosa"


def test_predict_rejects_bad_input(client):
    r = client.post("/predict", json={"instances": [{"sepal_length": "abc"}]})
    assert r.status_code == 422
