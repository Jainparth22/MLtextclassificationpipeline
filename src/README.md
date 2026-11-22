# 🐍 Source Scripts (`src/`)

This folder contains **Python scripts** extracted from the notebooks. `milestone_5_final.py` and the serving files run locally via `DATA_DIR`; the earlier milestone scripts are Kaggle-notebook exports with Kaggle data paths.

---

## Files

### `milestone_1_eda_preprocessing.py`
**Lines: ~979 | Sections: 22**

Complete EDA and baseline pipeline:
- Data loading with anti-leakage train/test separation
- Missing value analysis with custom `missing_report()` function
- Univariate analysis: triple-plot (histogram + box + Q-Q) for each numeric feature
- Bivariate analysis with Spearman correlation, Kruskal-Wallis tests, Cramér's V
- Text cleaning: `clean_text()` — lowercase, remove URLs, replace digits with `NUM`
- Feature engineering: vote ratios, identity flag aggregates, uppercase ratio, engagement features
- TF-IDF (15k, bigrams, `sublinear_tf=True`) + StandardScaler → hstack
- LogisticRegression baseline + 5-fold stratified CV
- Submission generation

---

### `milestone_2_linear_models_sgd.py`
**Lines: ~853 | Sections: 15**

Systematic linear model benchmark:
- Upgraded identity encoding (one-hot with `has_identity_data` binary)
- Log1p transforms for skewed features (if_1, if_2, upvote, downvote)
- Log1p transforms + StandardScaler (compared with MinMax; RobustScaler is discussed in the docs but not used in code)
- 4 TF-IDF configurations (A, B, C, D) evaluated systematically
- Models: LogReg, SGD (3 losses), Ridge, PassiveAggressive
- GridSearchCV and RandomizedSearchCV implementations
- SMOTE vs class_weight comparison with full analysis

---

### `milestone_3_nb_knn_svm.py`
**Lines: ~1064 | Sections: 18+**

Non-linear methods and feature engineering:
- TruncatedSVD with variance curve (shows 500 components = 27.2% variance)
- Chi-squared feature selection across K values: 10k, 20k, 30k, 40k, 50k, all
- MultinomialNB and ComplementNB with alpha tuning and MinMaxScaler for non-negativity
- KNN with 3 distance metrics (cosine, euclidean, manhattan) and k ∈ {3, 5, 7}
- LinearSVC C-value sweep: [0.01, 0.05, 0.08, 0.1, 0.15, 0.5]
- CalibratedClassifierCV (Platt scaling) for probability output
- 5-fold stratified CV confirming LinearSVC as champion

---

### `milestone_4_ensemble_final.py`
**Lines: ~1115 | Sections: 20+**

Ensemble methods with full experiment log:
- TruncatedSVD(300) → dense feature matrix for tree models
- GPU auto-detection: `nvidia-smi` → switch XGBoost to CUDA, LightGBM to GPU
- RandomForest, AdaBoost, XGBoost, LightGBM, MLP on dense features
- Custom class weight grid search (6 configurations)
- Stacking with `StackingClassifier` (4 base → LR meta, `cv=StratifiedKFold(3)`)
- Voting ensemble with soft probability averaging
- Seed averaging (5 seeds) using `decision_function` — not `predict`

---

### `milestone_5_final.py`
**The definitive submission script (the V5 "clean slate" pipeline)**

- Minimal text cleaning (no stemming, no lemmatization, no NLTK), ~17s
- 60k TF-IDF (35k word (1,3)-grams + 25k char_wb (3,5)-grams) + 14 numeric + 5 binary features
  (log1p on skewed columns, StandardScaler) = 60,019 dims
- Single 80/20 stratified validation split; `class_weight='balanced'` for the ensembles and
  custom weights `{0:1, 1:2, 2:1, 3:6}` for the stand-alone LinearSVC
- Stacking (4 base → LR meta), Voting and LinearSVC, retrained on ALL training rows
- Scored 0.826 Macro F1 on Kaggle, identical (0.82631) to the NLTK version, which showed
  lemmatization added nothing
- Reads data from `$DATA_DIR` (default `./data`)

### `milestone_5_v1_nltk_questions.py`
The M5 V1 script: the course questions Q1–Q10 plus the NLTK-lemmatized pipeline. Kept for the
record only (NLTK was not permitted by the competition rules).

---

### `toxic_pipeline.py`, `train_export.py`, `app.py`
Serving layer: one scikit-learn transformer that reproduces the V5 features from raw records,
a script that trains the V5 stack and exports a single `.joblib`, and a FastAPI app that serves it.
See "Serving the Model" in the top-level README.

### `hashing_sgd_experiment.py`
Out-of-core experiment: `HashingVectorizer` + `SGDClassifier.partial_fit` over CSV chunks. Not used
for the 0.826 score; see `docs/scalability_hashing.md`.

---

## Running Any Script

```bash
# milestone_5_final.py reads DATA_DIR (default ./data). The M1-M4 scripts still contain
# Kaggle paths (/kaggle/input/comment-category-prediction-challenge/...) near the top; edit them to run locally.

# Run:
DATA_DIR=data python src/milestone_5_final.py
```

**Runtime estimate:** ~3–4 hours on a machine with 16GB RAM.
