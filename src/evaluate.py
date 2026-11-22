"""
Leak-free evaluation, ablation, error analysis, fairness audit and model export.

Every step builds features INSIDE the training fold/split (FeatureBuilder is fit on training
rows only), fixing the validation leak of the original Kaggle notebooks, where TF-IDF and the
scaler were fit on all of train before splitting.

    python src/evaluate.py --step cv        --data-dir data --out-dir results/metrics
    python src/evaluate.py --step ablation  ...
    python src/evaluate.py --step holdout   ...   # stack vs LinearSVC, per-class, CI, errors, fairness, latency
    python src/evaluate.py --step export    ...   # fit on ALL rows, save joblib + metadata
    python src/evaluate.py --step report    ...   # collate everything into SUMMARY.md

Add --sample 20000 for a fast dry run (smaller vocabularies; numbers are NOT comparable to full runs).
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from toxic_pipeline import LABEL_NAMES, FeatureBuilder
from train_export import build_stack

CUSTOM_WEIGHTS = {0: 1, 1: 2, 2: 1, 3: 6}
SEED = 42


# ----------------------------------------------------------------------------- helpers
def load_train(args):
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv")).dropna(subset=["comment"])
    if args.sample:
        df = df.sample(args.sample, random_state=SEED)
    return df.reset_index(drop=True)


def make_fb(args, **kw):
    if args.sample:  # tiny vocabularies so dry runs stay fast
        kw = {"word_features": 3000, "char_features": 3000, "min_df": 2, **kw}
    return FeatureBuilder(**kw)


def lsvc(weights):
    return LinearSVC(C=0.1, class_weight=weights, max_iter=5000, random_state=SEED)


def per_class_f1(y, p):
    return f1_score(y, p, average=None, labels=[0, 1, 2, 3])


def mean_std(rows, col):
    v = np.array([r[col] for r in rows])
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


def save(out_dir, name, obj):
    path = os.path.join(out_dir, name)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    elif isinstance(obj, str):
        open(path, "w", encoding="utf-8").write(obj)
    else:
        json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, default=float)
    print(f"  wrote {path}")


def env_info():
    sha = "unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__, "joblib": joblib.__version__,
            "git_commit": sha, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def md_table(df, floatfmt=4):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- step: cv
def run_cv(args, configs, models, n_splits, tag):
    """Cross-validate (features fit inside each fold). configs: {name: FeatureBuilder kwargs}."""
    df = load_train(args)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for cfg_name, kw in configs.items():
        for k, (tr, va) in enumerate(skf.split(X, y)):
            t0 = time.time()
            fb = make_fb(args, **kw)
            Xtr = fb.fit_transform(X.iloc[tr], y[tr])
            Xva = fb.transform(X.iloc[va])
            t_feat = time.time() - t0
            for m_name, make_model in models.items():
                t1 = time.time()
                clf = make_model().fit(Xtr, y[tr])
                p = clf.predict(Xva)
                pc = per_class_f1(y[va], p)
                rows.append({"config": cfg_name, "model": m_name, "fold": k,
                             "macro_f1": f1_score(y[va], p, average="macro"),
                             "f1_c0": pc[0], "f1_c1": pc[1], "f1_c2": pc[2], "f1_c3": pc[3],
                             "n_features": Xtr.shape[1], "feat_s": round(t_feat, 1),
                             "fit_s": round(time.time() - t1, 1)})
                print(f"[{tag}] {cfg_name} | {m_name} | fold {k}: macroF1={rows[-1]['macro_f1']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    save(args.out_dir, f"{tag}_folds.csv", res)
    g = res.groupby(["config", "model"], sort=False)
    summ = g.agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                 f1_c0=("f1_c0", "mean"), f1_c1=("f1_c1", "mean"),
                 f1_c2=("f1_c2", "mean"), f1_c3=("f1_c3", "mean"),
                 n_features=("n_features", "first")).reset_index()
    save(args.out_dir, f"{tag}_summary.csv", summ)
    save(args.out_dir, f"{tag}_summary.md",
         f"### {tag} ({n_splits}-fold, features fit inside each fold)\n\n" + md_table(summ) + "\n")
    return summ


def step_cv(args):
    run_cv(args, {"final_features": {}},
           {"LinearSVC balanced": lambda: lsvc("balanced"),
            "LinearSVC custom {1:2,3:6}": lambda: lsvc(CUSTOM_WEIGHTS)}, 5, "cv")


def step_ablation(args):
    configs = {
        "word only": {"use_char": False, "use_structured": False},
        "char only": {"use_word": False, "use_structured": False},
        "word + char": {"use_structured": False},
        "word + char + structured (submitted)": {},
        "submitted + keep '*!?' in cleaning": {"clean_mode": "keep_punct"},
    }
    run_cv(args, configs, {"LinearSVC custom": lambda: lsvc(CUSTOM_WEIGHTS)}, 3, "ablation")


# ----------------------------------------------------------------------------- step: holdout
def bootstrap_ci(y, preds, n=1000, seed=0):
    """95% CI of macro-F1 for each model + paired difference vs the first model."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    scores = {k: [] for k in names}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        for k in names:
            scores[k].append(f1_score(y[idx], preds[k][idx], average="macro"))
    out = {}
    for k in names:
        lo, hi = np.percentile(scores[k], [2.5, 97.5])
        out[k] = {"macro_f1_ci95": [float(lo), float(hi)]}
    base = names[0]
    for k in names[1:]:
        d = np.array(scores[k]) - np.array(scores[base])
        out[k]["diff_vs_" + base] = {"mean": float(d.mean()),
                                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                    "p_better": float((d > 0).mean())}
    return out


def latency(fb, clf, X_raw, n_single=50):
    batch = X_raw.iloc[:1000]
    t0 = time.time()
    clf.predict(fb.transform(batch))
    batch_ms = (time.time() - t0) * 1000
    t0 = time.time()
    for i in range(n_single):
        clf.predict(fb.transform(X_raw.iloc[[i]]))
    single_ms = (time.time() - t0) * 1000 / n_single
    return {"ms_per_1000_batch": round(batch_ms, 1), "ms_per_single_request": round(single_ms, 2)}


def fairness_audit(val_raw, y, pred):
    """FPR = P(pred != 0 | true == 0), per identity group (only groups with >= 30 true-neutral rows)."""
    rows = []
    overall_fpr = float((pred[y == 0] != 0).mean())
    for col in ["race", "religion", "gender", "disability"]:
        if col not in val_raw.columns:
            continue
        g = val_raw[col].astype("object").where(val_raw[col].notna(), "missing").astype(str).to_numpy()
        for val in pd.unique(g):
            m = g == val
            neutral = m & (y == 0)
            if neutral.sum() < 30:
                continue
            rows.append({"attribute": col, "group": val, "n": int(m.sum()),
                         "n_true_neutral": int(neutral.sum()),
                         "fpr_neutral_flagged": float((pred[neutral] != 0).mean()),
                         "flagged_rate_all": float((pred[m] != 0).mean()),
                         "fpr_ratio_vs_overall": float((pred[neutral] != 0).mean() / max(overall_fpr, 1e-9))})
    return pd.DataFrame(rows).sort_values("fpr_ratio_vs_overall", ascending=False), overall_fpr


def step_holdout(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_train(args)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"])
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr_raw, Xva_raw = X.iloc[idx_tr], X.iloc[idx_va]
    ytr, yva = y[idx_tr], y[idx_va]

    t0 = time.time()
    fb = make_fb(args)
    Xtr = fb.fit_transform(Xtr_raw, ytr)
    Xva = fb.transform(Xva_raw)
    print(f"features: {Xtr.shape} in {time.time() - t0:.0f}s", flush=True)

    models, fit_s = {}, {}
    for name, clf in [("LinearSVC custom", lsvc(CUSTOM_WEIGHTS)),
                      ("Stacking (4 -> LR)", build_stack(args.n_jobs))]:
        t0 = time.time()
        models[name] = clf.fit(Xtr, ytr)
        fit_s[name] = round(time.time() - t0, 1)
        print(f"fit {name}: {fit_s[name]}s", flush=True)

    preds = {k: m.predict(Xva) for k, m in models.items()}
    stack = models["Stacking (4 -> LR)"]
    names = list(preds)

    result = {"n_train": int(len(ytr)), "n_val": int(len(yva)), "n_features": int(Xtr.shape[1]),
              "fit_seconds": fit_s, "env": env_info(), "models": {}}
    for k in names:
        p, r, f, s = precision_recall_fscore_support(yva, preds[k], labels=[0, 1, 2, 3], zero_division=0)
        result["models"][k] = {"macro_f1": float(f1_score(yva, preds[k], average="macro")),
                               "accuracy": float((preds[k] == yva).mean()),
                               "per_class": {LABEL_NAMES[c]: {"precision": float(p[c]), "recall": float(r[c]),
                                                              "f1": float(f[c]), "support": int(s[c])}
                                             for c in range(4)}}
    ci = bootstrap_ci(yva, preds, n=200 if args.sample else 1000)
    for k in names:
        result["models"][k].update(ci[k])
    result["latency"] = {k: latency(fb, m, Xva_raw) for k, m in models.items()}

    # confusion matrices
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    for ax, k in zip(np.atleast_1d(axes), names):
        cm = confusion_matrix(yva, preds[k], normalize="true", labels=[0, 1, 2, 3])
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    xticklabels=list(LABEL_NAMES.values()), yticklabels=list(LABEL_NAMES.values()))
        ax.set_title(f"{k}\nmacro-F1 {result['models'][k]['macro_f1']:.4f}")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrices.png"), dpi=130)
    plt.close(fig)

    # error analysis (stack): true class 3 that we miss + most common confusions
    proba = stack.predict_proba(Xva)
    pred = preds["Stacking (4 -> LR)"]
    miss = np.where((yva == 3) & (pred != 3))[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(miss, size=min(40, len(miss)), replace=False) if len(miss) else []
    err = pd.DataFrame({"true": 3, "predicted": pred[pick],
                        "p_class3": proba[pick, 3].round(3), "p_predicted": proba[pick].max(axis=1).round(3),
                        "n_words": Xva_raw["comment"].iloc[pick].astype(str).str.split().str.len().to_numpy(),
                        "comment": Xva_raw["comment"].iloc[pick].astype(str).str.slice(0, 400).to_numpy()})
    save(args.out_dir, "error_analysis_label3.csv", err)
    cm = confusion_matrix(yva, pred, labels=[0, 1, 2, 3])
    pairs = [(int(i), int(j), int(cm[i, j])) for i in range(4) for j in range(4) if i != j]
    result["top_confusions_stack"] = sorted(pairs, key=lambda t: -t[2])[:5]
    result["label3_missed"] = int(len(miss))
    result["label3_support"] = int((yva == 3).sum())

    # length effect on errors
    nw = Xva_raw["comment"].astype(str).str.split().str.len().to_numpy()
    bins = pd.qcut(nw, 4, duplicates="drop")
    result["accuracy_by_length_quartile_stack"] = {
        str(b): float((pred[bins == b] == yva[bins == b]).mean()) for b in pd.unique(bins)}

    # fairness
    fair, overall_fpr = fairness_audit(Xva_raw, yva, pred)
    save(args.out_dir, "fairness_audit.csv", fair)
    result["overall_fpr_neutral_flagged"] = overall_fpr

    np.save(os.path.join(args.out_dir, "val_predictions_stack.npy"), np.c_[idx_va, yva, pred])
    save(args.out_dir, "holdout.json", result)

    rows = [{"model": k, "macro_f1": result["models"][k]["macro_f1"],
             "ci95_lo": result["models"][k]["macro_f1_ci95"][0], "ci95_hi": result["models"][k]["macro_f1_ci95"][1],
             "accuracy": result["models"][k]["accuracy"], "fit_s": fit_s[k]} for k in names]
    pc = [{"model": k, "class": c, **{m: v for m, v in d.items() if m != "support"}, "support": d["support"]}
          for k in names for c, d in result["models"][k]["per_class"].items()]
    md = ["### Hold-out (80/20 stratified, features fit on train only)\n", md_table(pd.DataFrame(rows)),
          "\n\n#### Per-class\n", md_table(pd.DataFrame(pc)),
          "\n\n#### Stack vs LinearSVC (paired bootstrap)\n", "```\n" + json.dumps(
              result["models"]["Stacking (4 -> LR)"].get("diff_vs_LinearSVC custom"), indent=2) + "\n```",
          "\n\n#### Latency\n", "```\n" + json.dumps(result["latency"], indent=2) + "\n```",
          f"\n\nLabel-3 missed by stack: {result['label3_missed']} / {result['label3_support']}\n",
          "\n#### Fairness audit (false-positive rate on true-neutral comments)\n",
          f"Overall FPR: {overall_fpr:.4f}\n\n", md_table(fair.head(15)) if len(fair) else "(no groups large enough)"]
    save(args.out_dir, "holdout_summary.md", "\n".join(md))


# ----------------------------------------------------------------------------- step: export
def step_export(args):
    df = load_train(args)
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"])
    model = Pipeline([("features", make_fb(args)), ("clf", build_stack(args.n_jobs))])
    t0 = time.time()
    model.fit(X, y)
    print(f"trained on {len(y):,} rows in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, "final_stack_pipeline.joblib")
    joblib.dump(model, path, compress=3)

    # round-trip sanity check
    reloaded = joblib.load(path)
    chk = X.iloc[:200]
    assert (reloaded.predict(chk) == model.predict(chk)).all(), "round-trip predictions differ"

    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {**env_info(), "artifact": os.path.basename(path), "sha256": sha,
            "size_mb": round(os.path.getsize(path) / 1e6, 1), "n_train_rows": int(len(y)),
            "classes": [int(c) for c in model.classes_], "label_names": LABEL_NAMES,
            "sample_mode": bool(args.sample),
            "holdout_metrics": None}
    hp = os.path.join(args.out_dir, "holdout.json")
    if os.path.exists(hp) and not args.sample:
        h = json.load(open(hp))
        meta["holdout_metrics"] = {k: {"macro_f1": v["macro_f1"], "macro_f1_ci95": v["macro_f1_ci95"]}
                                   for k, v in h["models"].items()}
    save(args.model_dir, "model_metadata.json", meta)
    pins = [f"numpy=={meta['numpy']}", f"pandas=={meta['pandas']}", f"scipy=={meta['scipy']}",
            f"scikit-learn=={meta['sklearn']}", f"joblib=={meta['joblib']}",
            "fastapi==0.110.0", "uvicorn==0.29.0", "pydantic==2.6.4"]
    save(args.model_dir, "requirements-serve.lock.txt",
         f"# generated by evaluate.py --step export; python {meta['python']}\n" + "\n".join(pins) + "\n")
    print(f"saved {path} ({meta['size_mb']} MB, sha256 {sha[:12]}...)")


# ----------------------------------------------------------------------------- step: report
def step_report(args):
    parts = ["# Results summary (auto-generated)\n"]
    env = None
    hp = os.path.join(args.out_dir, "holdout.json")
    if os.path.exists(hp):
        env = json.load(open(hp))["env"]
    if env:
        parts.append("```\n" + json.dumps(env, indent=2) + "\n```\n")
    for f in ["cv_summary.md", "ablation_summary.md", "holdout_summary.md"]:
        p = os.path.join(args.out_dir, f)
        parts.append(open(p, encoding="utf-8").read() if os.path.exists(p) else f"(missing: {f})\n")
    save(args.out_dir, "SUMMARY.md", "\n\n".join(parts))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["cv", "ablation", "holdout", "export", "report"])
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--model-dir", default="src/model_artifacts")
    ap.add_argument("--sample", type=int, default=0, help="dry run on N random rows")
    ap.add_argument("--n-jobs", type=int, default=1, help="stacking CV workers (data is copied per worker)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    {"cv": step_cv, "ablation": step_ablation, "holdout": step_holdout,
     "export": step_export, "report": step_report}[args.step](args)


if __name__ == "__main__":
    main()
