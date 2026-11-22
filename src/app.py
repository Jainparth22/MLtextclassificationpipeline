"""
FastAPI service around the exported stacking pipeline.

    uvicorn app:app --app-dir src --port 8000          # then open http://localhost:8000/
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -H "X-API-Key: $KEY" -d '{"comment": "you are an idiot"}'

Only `comment` is required; the other fields are the competition's metadata columns and
default to "missing" values, exactly as during training.

Endpoints
    GET  /                web demo page
    GET  /health          liveness (open)
    GET  /info            artifact provenance           (API key if configured)
    GET  /metrics         Prometheus text metrics       (API key if configured)
    POST /predict         one comment                   (API key + rate limit)
    POST /predict/batch   1..500 comments               (API key + rate limit)

Config via environment: MODEL_PATH, API_KEYS, RATE_LIMIT_PER_MIN, TRUST_PROXY, LOG_LEVEL
(see api_support.py). Comment text is never logged.
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from api_support import (Metrics, RateLimiter, check_key, client_id, configure_logging,
                         load_api_keys, log_event, raise_unauthorized)
from toxic_pipeline import LABEL_NAMES
from web_ui import CSP, INDEX_HTML

MODEL_PATH = os.environ.get("MODEL_PATH", "src/model_artifacts/final_stack_pipeline.joblib")
MAX_BATCH = 500

state = {}
metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Create one with "
            "`python src/evaluate.py --step export` or set MODEL_PATH.")
    state["model"] = joblib.load(MODEL_PATH)
    meta_path = os.path.join(os.path.dirname(MODEL_PATH), "model_metadata.json")
    state["meta"] = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    state["api_keys"] = load_api_keys()
    state["limiter"] = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_MIN", "60")))
    log_event(event="startup", auth_enabled=bool(state["api_keys"]),
              rate_limit_per_min=state["limiter"].per_minute, model=os.path.basename(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title="Toxic Comment Classifier", version="1.1", lifespan=lifespan)


# ------------------------------------------------------------------ middleware
@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    request.state.labels = None
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        log_event(event="unhandled_error", request_id=rid, path=request.url.path)
        raise
    finally:
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.observe_request(request.method, path, status, dt)
        log_event(event="request", request_id=rid, method=request.method, path=path, status=status,
                  ms=round(dt * 1000, 1), client=client_id(request, None),
                  labels=request.state.labels)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ------------------------------------------------------------------ guards
def require_key(request: Request, x_api_key: Optional[str] = Header(default=None)):
    """API-key check (if keys are configured) for protected endpoints."""
    if not check_key(x_api_key, state.get("api_keys", [])):
        metrics.inc("auth_failures")
        raise_unauthorized()
    return x_api_key


def guard_predict(request: Request, x_api_key: Optional[str] = Depends(require_key)):
    """Auth + per-caller rate limit for prediction endpoints."""
    ok, retry = state["limiter"].check(client_id(request, x_api_key))
    if not ok:
        metrics.inc("rate_limited")
        raise HTTPException(status_code=429, detail="rate limit exceeded",
                            headers={"Retry-After": str(retry)})


# ------------------------------------------------------------------ schemas
class Comment(BaseModel):
    comment: str = Field(..., max_length=20000)
    created_date: Optional[str] = None
    upvote: int = 0
    downvote: int = 0
    if_1: float = 0
    if_2: float = 0
    emoticon_1: int = 0
    emoticon_2: int = 0
    emoticon_3: int = 0
    disability: bool = False
    race: Optional[str] = None
    religion: Optional[str] = None
    gender: Optional[str] = None


class Prediction(BaseModel):
    label: int
    label_name: str
    probabilities: dict


def _predict(items: List[Comment], request: Request) -> List[Prediction]:
    model = state["model"]
    df = pd.DataFrame([i.model_dump() for i in items])
    proba = model.predict_proba(df)
    classes = model.classes_
    out = []
    for row in proba:
        best = int(classes[row.argmax()])
        out.append(Prediction(
            label=best, label_name=LABEL_NAMES[best],
            probabilities={LABEL_NAMES[int(c)]: round(float(p), 4) for c, p in zip(classes, row)}))
    names = [o.label_name for o in out]
    metrics.observe_predictions(names)
    request.state.labels = names if len(names) <= 5 else {n: names.count(n) for n in set(names)}
    return out


# ------------------------------------------------------------------ routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return HTMLResponse(INDEX_HTML, headers={"Content-Security-Policy": CSP})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/info", dependencies=[Depends(require_key)])
def info():
    """Provenance of the loaded artifact (versions, training date, hold-out metrics, sha256)."""
    return state.get("meta", {})


@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_key)])
def prometheus_metrics():
    return metrics.render()


@app.post("/predict", response_model=Prediction, dependencies=[Depends(guard_predict)])
def predict(item: Comment, request: Request):
    return _predict([item], request)[0]


@app.post("/predict/batch", response_model=List[Prediction], dependencies=[Depends(guard_predict)])
def predict_batch(items: List[Comment], request: Request):
    if not 0 < len(items) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"send between 1 and {MAX_BATCH} items")
    return _predict(items, request)
