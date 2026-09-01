# 🎓 Viva Preparation (`viva_prep/`)

This folder contains my **oral examination preparation** for the project — structured Q&A on every algorithm, design decision, and technique used.

Including this folder in the repository is intentional. It demonstrates something rare: **the ability to explain not just what was implemented, but precisely why**, with mathematical backing.

Most ML practitioners can write the code. Fewer can answer:
- *"Why did you use Kruskal-Wallis instead of ANOVA for your EDA?"*
- *"Mathematically, why does TruncatedSVD hurt LinearSVC even though it reduces noise?"*
- *"What's the difference between L1 and L2 regularisation for your specific feature space?"*
- *"Why is your meta-learner a LogisticRegression and not XGBoost?"*

The answers are all in `viva_guide.md`.

---

## `viva_guide.md`

A complete Q&A reference covering:

1. **EDA** — visualization techniques (violin, Q-Q, heatmap), correlation types (Pearson vs Spearman vs Cramér's V), statistical tests (Kruskal-Wallis, Chi-squared, Shapiro-Wilk)
2. **Preprocessing** — all missing value strategies with rationale, every text cleaning technique tested and its result
3. **Feature engineering** — every engineered feature with formula and expected signal
4. **Feature transformation** — log1p, StandardScaler, RobustScaler, TF-IDF — when and why
5. **Feature selection** — TF-IDF max_features vs Chi² vs TruncatedSVD — actual results
6. **Class imbalance** — all 5 approaches tested (class_weight, SMOTE, ROS, threshold tuning, custom weights) with exact Kaggle scores
7. **Model journey** — M1 through M5 with key learnings at each step

This was prepared for a 2-level viva examination where faculty challenged every design choice with follow-up questions. The answers here are the actual responses that passed those examinations.
