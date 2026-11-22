import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tiny_model, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", tiny_model)
    import app
    importlib.reload(app)
    with TestClient(app.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_shape_and_probabilities(client):
    r = client.post("/predict", json={"comment": "kill you idiot scum"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in (0, 1, 2, 3)
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01


def test_batch_and_limits(client):
    ok = client.post("/predict/batch", json=[{"comment": "nice great"}, {"comment": "stupid hate"}])
    assert ok.status_code == 200 and len(ok.json()) == 2
    assert client.post("/predict/batch", json=[]).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_info_endpoint_exists(client):
    assert client.get("/info").status_code == 200
