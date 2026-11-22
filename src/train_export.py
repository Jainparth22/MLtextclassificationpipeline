"""
Train the submitted V5 stacking model end-to-end and export ONE joblib artifact
(feature builder + classifier) for serving with src/app.py.

    python src/train_export.py --data-dir data --out src/model_artifacts/final_stack_pipeline.joblib

Full training on all 197,999 rows is slow (hours; the stack is 4 base models x 3-fold CV)
and memory hungry (~16 GB). Use --sample 5000 for a quick smoke test of the plumbing
(the resulting model is NOT the 0.826 model).
"""
import argparse
import os
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from toxic_pipeline import FeatureBuilder


def build_stack(n_jobs: int) -> StackingClassifier:
    """Same hyper-parameters as the submitted V5 stack (class_weight='balanced')."""
    return StackingClassifier(
        estimators=[
            ("cal_lsvc", CalibratedClassifierCV(
                LinearSVC(C=0.1, class_weight="balanced", max_iter=5000, random_state=42),
                cv=3, method="sigmoid")),
            ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
                                      class_weight="balanced", random_state=42)),
            ("sgd", SGDClassifier(loss="log_loss", alpha=1e-05, penalty="l2",
                                  class_weight="balanced", max_iter=3000,
                                  random_state=42)),
            ("pac", CalibratedClassifierCV(
                PassiveAggressiveClassifier(C=0.1, class_weight="balanced",
                                            max_iter=3000, random_state=42),
                cv=3, method="sigmoid")),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=3000, random_state=42),
        cv=3, n_jobs=n_jobs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--out", default="src/model_artifacts/final_stack_pipeline.joblib")
    ap.add_argument("--sample", type=int, default=0, help="train on a random subset (smoke test)")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="stacking CV workers; -1 copies the data per worker and caused OOM in M4")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv")).dropna(subset=["comment"])
    if args.sample:
        train = train.sample(args.sample, random_state=42)
    y = train["label"].to_numpy()
    X = train.drop(columns=["label"])

    # smaller vocabularies for tiny samples so min_df / max_features stay meaningful
    fb = FeatureBuilder() if not args.sample else FeatureBuilder(2000, 2000, min_df=2)
    model = Pipeline([("features", fb), ("clf", build_stack(args.n_jobs))])

    t0 = time.time()
    model.fit(X, y)
    print(f"trained on {len(y):,} rows in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(model, args.out, compress=3)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
