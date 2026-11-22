# Deployment & Model Management

## 1. Model artifact
* One file: `final_stack_pipeline.joblib` = `Pipeline(FeatureBuilder -> StackingClassifier)`.
  Created by `python src/evaluate.py --step export` (the Colab notebook does this).
* Next to it: `model_metadata.json` (python/sklearn/numpy/pandas/scipy/joblib versions, git commit, UTC date,
  sha256, size, hold-out metrics) and `requirements-serve.lock.txt` (exact pins for serving).
* **Pickles are bound to library versions** and execute code on load: serve with the pinned versions and only load
  files you produced. (`skops` is a safer serialisation option for sklearn if you ever need to accept third-party models.)
* Keep large artifacts out of git. Options: GitHub Release asset, Hugging Face Hub model repo, Azure Blob Storage.
  Verify the `sha256` after download.

After the Colab run:
```bash
cp model_artifacts/requirements-serve.lock.txt requirements-serve.txt    # exact pins from training
echo "3.12" > .python-version                                            # the Python in model_metadata.json
cp model_artifacts/* src/model_artifacts/
```

## 2. Local
```bash
pip install -r requirements-serve.txt
uvicorn app:app --app-dir src --port 8000
```

## 3. Container
```bash
docker build --build-arg PYTHON_VERSION=3.12 -t toxic-api .
docker run -p 8000:8000 toxic-api
curl localhost:8000/health ; curl localhost:8000/info
```

## 4. Free public demo: Hugging Face Space (Docker SDK)
1. Create a Space (SDK: Docker) and push the `Dockerfile`, `src/` and `requirements-serve.txt`.
2. Do not commit the model; download it from your HF model repo at start-up with
   `huggingface_hub.hf_hub_download` and set `MODEL_PATH`.
3. Put the Space URL in the README.

## 5. Azure path (relevant for Microsoft roles)
* Registry: `az acr build -r <acr> -t toxic-api:1 .`
* Run: **Azure Container Apps** (`az containerapp up --image <acr>.azurecr.io/toxic-api:1 --target-port 8000 --ingress external`)
  or an **Azure ML managed online endpoint** for traffic splitting and blue/green rollout.
* Artifact in Blob Storage, loaded at start-up with a managed identity (no keys in the image).
* Observability: Application Insights (latency, failures) plus a log of the predicted-class mix.
* CI/CD: GitHub Actions already runs tests; add build-and-push on tags.

## 6. API hardening already built in
`API_KEYS` (constant-time comparison, `401` + `WWW-Authenticate`), per-caller sliding-window rate limit (`429` +
`Retry-After`), request IDs, JSON logs without comment text, Prometheus `/metrics`, CSP on the demo page.
In Azure, inject `API_KEYS` from Key Vault (Container Apps secrets) rather than baking it into the image, and scrape
`/metrics` or rely on Application Insights. Known gap: rate limiting is per process.

## 7. Production concerns worth being able to discuss
| Topic | Position |
|---|---|
| Latency/cost | The stack runs 4 models per request. Measure it (`holdout_summary.md`); if its CI over LinearSVC includes 0, ship LinearSVC (+ calibration) |
| Throughput | `/predict/batch` vectorises; run N workers (`uvicorn --workers`); the model is read-only but memory is per worker |
| Safety | Batch/body limits, no logging of raw text by default, human-in-the-loop for any action |
| Monitoring | Input drift (length, OOV rate), class mix, sampled human labels for class-3 recall, identity-group FPR |
| Retraining | Scheduled and on drift; compare candidate vs current on a frozen hold-out with a paired bootstrap before promoting |
| Scale to 10M+ rows | HashingVectorizer + `partial_fit` (see `docs/scalability_hashing.md`) |
| Better accuracy | Fine-tuned transformer; distil or use ONNX for latency (see the DistilBERT notebook) |
