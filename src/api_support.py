"""
Operational helpers for the API: API-key auth, rate limiting, Prometheus-style metrics, JSON logging.
Dependency-free on purpose (stdlib + FastAPI only).

Environment variables (read at start-up):
    API_KEYS            comma-separated keys. Empty/unset = auth disabled (local/demo mode).
    RATE_LIMIT_PER_MIN  requests per minute per caller on /predict*. 0 disables. Default 60.
    TRUST_PROXY         "1" = take the client IP from X-Forwarded-For (only behind a proxy you control).
    LOG_LEVEL           default INFO.
"""
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, Request

logger = logging.getLogger("toxic_api")


def configure_logging():
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))  # message is already JSON
        logger.addHandler(h)
        logger.propagate = False
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


def log_event(**fields):
    """One JSON object per line. Never pass raw comment text here (privacy)."""
    logger.info(json.dumps(fields, default=str))


def load_api_keys() -> List[str]:
    return [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]


def client_id(request: Request, api_key: Optional[str]) -> str:
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:10]
    ip = request.client.host if request.client else "unknown"
    if os.environ.get("TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            ip = fwd.split(",")[0].strip()
    return "ip:" + ip


class RateLimiter:
    """Sliding-window limiter, in memory. NOTE: per process; with N workers the effective limit is N x limit.
    Use a shared store (Redis) or the gateway's limiter for a multi-instance deployment."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def check(self, who: str, now: Optional[float] = None) -> Tuple[bool, int]:
        if self.per_minute <= 0:
            return True, 0
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[who]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, max(1, int(60 - (now - q[0])) + 1)
            q.append(now)
            self._calls += 1
            if self._calls % 1000 == 0:  # drop idle callers so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= 60]:
                    del self._hits[k]
            return True, 0


class Metrics:
    """Minimal Prometheus text-format metrics."""
    BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = defaultdict(int)          # (method, path, status) -> n
        self.predictions = defaultdict(int)       # label_name -> n
        self.rate_limited = 0
        self.auth_failures = 0
        self.lat_buckets = defaultdict(lambda: [0] * len(self.BUCKETS))
        self.lat_sum = defaultdict(float)
        self.lat_count = defaultdict(int)

    def observe_request(self, method, path, status, seconds):
        with self._lock:
            self.requests[(method, path, status)] += 1
            self.lat_sum[path] += seconds
            self.lat_count[path] += 1
            for i, b in enumerate(self.BUCKETS):
                if seconds <= b:
                    self.lat_buckets[path][i] += 1

    def observe_predictions(self, label_names):
        with self._lock:
            for n in label_names:
                self.predictions[n] += 1

    def inc(self, name):
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def render(self) -> str:
        with self._lock:
            out = ["# TYPE toxic_http_requests_total counter"]
            for (m, p, s), n in sorted(self.requests.items()):
                out.append(f'toxic_http_requests_total{{method="{m}",path="{p}",status="{s}"}} {n}')
            out.append("# TYPE toxic_predictions_total counter")
            for name, n in sorted(self.predictions.items()):
                out.append(f'toxic_predictions_total{{label="{name}"}} {n}')
            out += ["# TYPE toxic_rate_limited_total counter", f"toxic_rate_limited_total {self.rate_limited}",
                    "# TYPE toxic_auth_failures_total counter", f"toxic_auth_failures_total {self.auth_failures}",
                    "# TYPE toxic_http_request_duration_seconds histogram"]
            for path in sorted(self.lat_count):
                for b, c in zip(self.BUCKETS, self.lat_buckets[path]):
                    out.append(f'toxic_http_request_duration_seconds_bucket{{path="{path}",le="{b}"}} {c}')
                out.append(f'toxic_http_request_duration_seconds_bucket{{path="{path}",le="+Inf"}} {self.lat_count[path]}')
                out.append(f'toxic_http_request_duration_seconds_sum{{path="{path}"}} {self.lat_sum[path]:.6f}')
                out.append(f'toxic_http_request_duration_seconds_count{{path="{path}"}} {self.lat_count[path]}')
            return "\n".join(out) + "\n"


def check_key(provided: Optional[str], keys: List[str]) -> bool:
    if not keys:
        return True
    return bool(provided) and any(hmac.compare_digest(provided.encode(), k.encode()) for k in keys)


def raise_unauthorized():
    raise HTTPException(status_code=401, detail="invalid or missing API key",
                        headers={"WWW-Authenticate": "ApiKey"})
