# Decision Log

Format: decision -> alternatives -> evidence -> outcome / caveat. Numbers are Kaggle-leaderboard or single-split
unless marked "CV" (see `results/metrics/SUMMARY.md` for leak-free CV figures).

| # | Decision | Alternatives considered | Evidence | Outcome / caveat |
|---|---|---|---|---|
| 1 | Optimise **Macro-F1**, use `class_weight` | accuracy, SMOTE, oversampling | M1 accuracy 0.89 hid class-3 recall ~0.40; oversampling + weights double-corrected | Class weights kept. Weights tuned per run (see `results/all_experiments_summary.md`) |
| 2 | **Sparse TF-IDF + linear models**, no dimensionality reduction | SVD + trees/MLP, KNN | SVD 200-500 comps kept ~18-27% variance; trees on SVD 0.75-0.80, KNN ~0.56 | Linear on sparse won clearly (M3 +0.04) |
| 3 | **LinearSVC** as workhorse | LogReg (lbfgs), SGD | similar/better F1, 3-5x faster fit | `C=0.1`; grid over C moved F1 by < 0.001 (within noise) |
| 4 | Add **char_wb 3-5-grams** | word-only | gain in M3/M5 | Caveat: V5 cleaning strips `*`, so masked profanity is split. The ablation tests keeping `*!?` |
| 5 | **No stop-word removal** | `stop_words='english'` | slightly lower score with removal | Plausible mechanism: "you/not/are" carry meaning; one split, no CI at the time |
| 6 | **No lemmatization/stemming** | NLTK lemmatizer, regex stemmer | NLTK: identical score to 5 dp; regex stemmer hurt | NLTK also wasn't permitted |
| 7 | **Stacking (LR meta)** over single LinearSVC | voting, LightGBM meta | +0.003-0.004 on leaderboard; LightGBM meta 0.787 | Gain is small vs ~8x inference cost; hold-out step measures both |
| 8 | **LR meta-learner** | LightGBM/XGBoost | only 16 inputs, so a nonlinear meta overfits | |
| 9 | Dropped **SMOTE, pseudo-labeling, SVD, threshold tuning, seed averaging** | - | each hurt or gave nothing in single-split runs | The Chi² "failure" was a bug (`abs()` on scaled data), not a finding |
| 10 | **Fit features inside folds** (this repo) | fit on all train (original) | methodological | Removes a validation leak; expect numbers within noise of the original |
| 11 | One **`Pipeline(FeatureBuilder -> clf)`** artifact | save vectorizer/scaler/model separately | train-serve skew is the usual failure | Versions recorded in `model_metadata.json` |
| 12 | Ship the **stack** or LinearSVC? | | latency and CI in `holdout_summary.md` | Decide with numbers: if the paired-bootstrap CI of the difference includes 0, prefer the cheaper model |
| 13 | **Bias audit** by identity group | skip | Macro-F1 is blind to who is mis-flagged | `fairness_audit.csv` |
| 14 | **DistilBERT** as post-competition extension | none (banned in competition) | quantifies the cost of the constraint | `colab_distilbert_extension.ipynb` |

## Things I would do next
1. Keep masking characters in cleaning if the ablation supports it, and re-tune on CV.
2. Calibrate the final probabilities on a held-out set (reliability curve) and choose per-class operating thresholds
   from a cost model, not argmax.
3. Distil the transformer into a small model or use ONNX Runtime for latency.
4. Add drift monitoring and a human-labelled sample for ongoing class-3 recall.
