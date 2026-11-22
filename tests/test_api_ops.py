import importlib
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api_support import RateLimiter, check_key


def make_client(tiny_model, monkeypatch, **env):
    monkeypatch.setenv("MODEL_PATH", tiny_model)
    for k in ("API_KEYS", "RATE_LIMIT_PER_MIN", "TRUST_PROXY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app
    importlib.reload(app)
    return TestClient(app.app)


# ---------------------------------------------------------------- unit
def test_rate_limiter_sliding_window():
    rl = RateLimiter(3)
    assert [rl.check("a", now=t)[0] for t in (0, 1, 2)] == [True, True, True]
    ok, retry = rl.check("a", now=3)
    assert not ok and retry >= 1
    assert rl.check("b", now=3)[0]          # other callers unaffected
    assert rl.check("a", now=61)[0]         # window slid


def test_rate_limiter_disabled():
    assert RateLimiter(0).check("x")[0]


def test_check_key():
    assert check_key(None, [])              # no keys configured -> open
    assert check_key("k1", ["k1", "k2"])
    assert not check_key("bad", ["k1"]) and not check_key(None, ["k1"])


# ---------------------------------------------------------------- integration
def test_auth_required_when_keys_set(tiny_model, monkeypatch):
    with make_client(tiny_model, monkeypatch, API_KEYS="secret1,secret2") as c:
        body = {"comment": "hello there"}
        assert c.post("/predict", json=body).status_code == 401
        assert c.post("/predict", json=body, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/predict", json=body, headers={"X-API-Key": "secret2"}).status_code == 200
        assert c.get("/info").status_code == 401
        assert c.get("/health").status_code == 200       # stays open
        assert c.get("/").status_code == 200             # UI stays open


def test_open_when_no_keys(tiny_model, monkeypatch):
    with make_client(tiny_model, monkeypatch) as c:
        assert c.post("/predict", json={"comment": "hi"}).status_code == 200


def test_rate_limit_429_with_retry_after(tiny_model, monkeypatch):
    with make_client(tiny_model, monkeypatch, RATE_LIMIT_PER_MIN="2") as c:
        codes = [c.post("/predict", json={"comment": "hi"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = c.post("/predict", json={"comment": "hi"})
        assert int(r.headers["Retry-After"]) >= 1
        assert c.get("/health").status_code == 200       # not rate limited


def test_metrics_endpoint(tiny_model, monkeypatch):
    with make_client(tiny_model, monkeypatch) as c:
        c.post("/predict", json={"comment": "nice great"})
        text = c.get("/metrics").text
        assert 'toxic_http_requests_total{method="POST",path="/predict",status="200"} 1' in text
        assert "toxic_predictions_total" in text
        assert "toxic_http_request_duration_seconds_count" in text


def test_request_id_and_security_headers(tiny_model, monkeypatch):
    with make_client(tiny_model, monkeypatch) as c:
        r = c.get("/health", headers={"X-Request-ID": "abc123"})
        assert r.headers["X-Request-ID"] == "abc123"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Request-ID" in c.get("/health").headers


def test_logs_never_contain_comment_text(tiny_model, monkeypatch):
    secret = "my-very-private-sentence-xyz"
    with make_client(tiny_model, monkeypatch) as c:
        logger = logging.getLogger("toxic_api")
        records = []
        handler = logging.Handler()
        handler.emit = lambda rec: records.append(rec.getMessage())
        logger.addHandler(handler)
        try:
            c.post("/predict", json={"comment": secret})
        finally:
            logger.removeHandler(handler)
    assert records, "expected a request log line"
    assert all(secret not in r for r in records)
    line = json.loads([r for r in records if '"request"' in r][-1])
    assert line["path"] == "/predict" and line["status"] == 200 and line["labels"]


def test_web_ui_served_with_csp(tiny_model, monkeypatch):
    with make_client(tiny_model, monkeypatch) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Toxic Comment Classifier" in r.text
        assert "default-src 'none'" in r.headers["Content-Security-Policy"]
