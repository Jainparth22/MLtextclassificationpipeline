# 📚 Documentation (`docs/`)

This folder contains **deep technical documentation** for every milestone. Unlike the notebooks (which show *what* was done), these documents explain *why* every decision was made — the mathematical reasoning, the alternatives considered, and the examiner-ready justifications.

This level of documentation is rare in student ML projects. It reflects a commitment to understanding algorithms at their foundations, not just calling `.fit()`.

---

## Milestone Explanations

### `milestone_1_explanation.md`
Cell-by-cell breakdown of the EDA and baseline notebook (627 lines).

**Covers:**
- Why `sample()` over `head()` for unbiased data inspection
- Why Spearman over Pearson (non-normal distributions)
- Why Kruskal-Wallis over ANOVA (non-parametric — our features violate normality)
- Why frequency encoding for `post_id` (avoids OHE dimensionality explosion)
- The anti-leakage rule: fit TF-IDF on training split ONLY
- Why Macro F1 over accuracy (20.9:1 imbalance)
- Why `sublinear_tf=True` (logarithmic term frequency — diminishing returns)
- Why median imputation not mean (robust to vote outliers)
- Every TF-IDF hyperparameter with mathematical justification
- 7 viva Q&A with model answers

---

### `milestone_2_explanation.md`
Cell-by-cell breakdown of the linear models comparison (386 lines).

**Covers:**
- Why RobustScaler (median/IQR) over StandardScaler for our specific outlier profile
- Why character n-grams capture censored profanity and misspellings
- The `has_identity_data` binary — why missing demographic data IS a signal
- SGD loss functions: log_loss vs hinge vs modified_huber — mathematical differences
- Elastic Net (L1+L2) — when to use which regularisation
- Why SMOTE must be applied AFTER train-val split (data leakage prevention)
- GridSearchCV vs RandomizedSearchCV — when each is appropriate
- The golden rule: any transformation that "learns" must ONLY see training data
- 6 detailed viva Q&A

---

### `milestone_3_explanation.md`
Cell-by-cell breakdown of the NB + KNN + SVM notebook (510 lines).

**Covers:**
- TruncatedSVD vs PCA — why PCA is impossible on sparse matrices (93GB RAM)
- Latent Semantic Analysis (LSA) — mathematical intuition
- Why L2 normalisation after SVD
- Chi² feature selection — formula, why it works for TF-IDF, actual results per K
- MultinomialNB vs ComplementNB — the imbalanced dataset argument
- Why KNN fails in 35k-dimensional space (curse of dimensionality — quantified)
- LinearSVC vs SVC(kernel='linear') — liblinear vs libsvm complexity
- CalibratedClassifierCV: Platt scaling (sigmoid) vs isotonic regression
- Why SVD hurts LinearSVC — variance ≠ discrimination (rare words have low variance but high signal)
- 10 detailed viva Q&A

---

### `milestone_4_explanation.md`
Explanation of ensemble methods and why trees failed (187 lines).

**Covers:**
- Why tree models failed on SVD-dense features (82% variance lost = degraded signal)
- Stacking vs Voting — mathematical difference (learned combination vs fixed weights)
- Why these 4 specific base models (diversity of loss functions: hinge, log, online SGD, PA)
- Custom class weight formula vs sklearn's balanced formula
- Pseudo-labeling theory and why 95% confidence threshold is necessary
- Seed averaging: why use `decision_function` not `predict` (averaging discrete labels is meaningless)
- `passthrough=False` in StackingClassifier — why we don't give the meta-learner raw features

---

### `milestone_5_explanation.md`
The final milestone — what the consolidated notebook contains (146 lines).

**Covers:**
- V5-style minimal cleaning vs aggressive cleaning — why less is more
- Why LinearSVC over LogReg (O(n×d) vs O(n×d²))
- Why custom weights over balanced
- Why seed averaging instead of stacking for the final submission
- Why 60k features is the sweet spot (80k caused OOM on Kaggle)
- The honest answer to "What would you need to hit 0.87+?" — BERT, contextual embeddings

---

## `output_insights/`

Actual model output analysis from notebook runs — specific numbers, confusion matrices, and what the results revealed:

- **`milestone_1_output_insights.md`** — Baseline model performance breakdown, class-by-class F1, confusion matrix analysis
- **`milestone_2_output_insights.md`** — All 4 models compared, scaler experiment results, SMOTE vs balanced findings
- **`milestone_3_output_insights.md`** — Chi² K-value results, SVD variance table, LinearSVC C-sweep results

---

## `project_retrospective.md`

**The complete research logbook** — everything that happened, why, and what was learned.

This is the most honest ML project document you'll read. It includes:
- Every version of M5 (V1 through V7) with exact Kaggle scores
- Technical bugs encountered and exactly how they were fixed
- What failed and the specific reasoning for each failure
- The leaderboard context (our 0.826 vs top 10's 0.845)
- 10 key lessons learned
- Future directions if the library constraints were lifted
- The philosophical conclusion: 0.826 is not a failure — it's an empirically validated ceiling for TF-IDF + sklearn linear models on this dataset
