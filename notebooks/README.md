# 📓 Notebooks

This folder contains **all Jupyter notebooks** for the project, organised chronologically by milestone. Each notebook represents a complete, self-contained experiment that was submitted to Kaggle.

---

## Milestone Notebooks

### `milestone_1_eda_baseline.ipynb`
**Kaggle Score: ~0.777 Macro F1**

The foundation of the project. Contains:
- **22-section EDA** — class distribution, missing value analysis (race/religion/gender ~75-85% missing), correlation heatmaps, violin plots, word clouds per class, temporal analysis, engagement feature analysis
- **Feature engineering** — `comment_length`, `word_count`, `vote_ratio`, `has_detection`
- **Baseline model** — TF-IDF (15k bigrams) + LogisticRegression(C=2.0, balanced) with `ColumnTransformer` pipeline
- **Key finding** — Label 3 (2.76% of data) had very low recall (~0.40) — identified as the core bottleneck

---

### `milestone_2_linear_models.ipynb`
**Kaggle Score: ~0.776 | Val F1: 0.809 (PA model)**

Systematic linear model comparison:
- **4 TF-IDF configurations** — word(15k), word(25k+trigrams), word+char(10k), word(25k)+char(10k)
- **3 scaler experiments** — StandardScaler, MinMaxScaler, Log1p+StandardScaler
- **Models compared** — LogisticRegression, SGDClassifier (3 loss functions), RidgeClassifier, PassiveAggressiveClassifier
- **SMOTE vs class_weight** — SMOTE hurt SGD; class_weight won
- **Key innovation** — precomputing TF-IDF once outside GridSearchCV reduced runtime from ~2 hours to 15 minutes

---

### `milestone_3_nb_knn_svm.ipynb`
**Kaggle Score: 0.816 (+0.04 — biggest single jump)**

Explores non-linear and kernel methods:
- **TruncatedSVD (LSA)** — 500 components = only 27.2% variance → proved SVD hurts LinearSVC
- **Chi-squared feature selection** — K=50k marginally best; all features nearly equally informative
- **MultinomialNB & ComplementNB** — both ~0.55-0.65 F1; independence assumption too strong
- **KNN** — 0.59 F1 even after SVD reduction; worst model tested; curse of dimensionality
- **LinearSVC** — 0.826 val F1 with C=0.08-0.1, class_weight='balanced' → **new champion**
- **Key finding** — LinearSVC on full 60k sparse features (0.826) vs SVD-reduced (much lower)

---

### `milestone_4_ensemble.ipynb`
**Kaggle Score: 0.823**

Ensemble methods and tree-based models:
- **Tree models on SVD-dense features** — RF (0.78), XGBoost (0.80), LightGBM (0.80), MLP (0.82)  
  → All failed vs LinearSVC because SVD loses 80%+ of discriminative information
- **Stacking ensemble** — 4 linear base models → LR meta-learner → **0.826 val F1**
- **Voting ensemble** — soft voting of top 4 → 0.825 val F1
- **Custom class weights search** — `{0:1, 1:4.5, 2:2, 3:12}` beat sklearn's auto-balanced

---

## `milestone_5/` — The Final Push (6 Versions)

| File | Strategy | Kaggle Score | Outcome |
|---|---|:---:|---|
| `v1_feature_expansion.ipynb` | 45k TF-IDF + trigrams + NLTK lemma | 0.826 | Baseline for M5 |
| `v3_hybrid_strategy.ipynb` | Hybrid class weights + 60k TF-IDF | 0.826 | Same as V1 |
| `v4_max_attempt.ipynb` | Everything: SMOTE+ROS+LightGBM+pseudo-labeling | OOM crash | ❌ All additions hurt |
| **`v5_clean_final.ipynb`** | **Stripped back to the recipe that held up** | **0.826** | **⭐ SUBMITTED** |
| `v6_diversity_experiment.ipynb` | ComplementNB + LightGBM meta-learner | 0.787 | ❌ Worse — diversity ≠ quality |
| `v7_final_tuning.ipynb` | Chi² + GridSearch C + seed averaging | 0.827 (val) | Marginal; same Kaggle |

**V5 is the definitive version** — it proved that NLTK lemmatization added zero value (identical score to 5 decimal places: 0.82631) and that the simplest clean pipeline is optimal.


---

## Colab notebooks (final evaluation)

| Notebook | Purpose |
|---|---|
| `colab_final_evaluation.ipynb` | Leak-free 5-fold CV, ablation, hold-out stack vs LinearSVC with bootstrap CIs, error analysis, fairness audit, model export. Outputs go to `results/metrics/`. |
| `colab_distilbert_extension.ipynb` | Post-competition DistilBERT fine-tune on the same split (GPU) for the accuracy/latency/size trade-off. |

The `milestone_*` and `milestone_5/` notebooks are the original Kaggle runs (training evidence). Add the public
Kaggle URL of each version here when available.
