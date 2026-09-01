# Milestone 1 — Output Summary, Insights & Strategy for Next Milestones

> This document summarises every key finding from the executed Milestone 1 notebook on Kaggle, explains **why** each observation matters, and lays out a concrete strategy for Milestones 2–5.

---

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Missing Value Analysis](#2-missing-value-analysis)
3. [Data Types — Critical Discoveries](#3-data-types--critical-discoveries)
4. [Duplicate Analysis](#4-duplicate-analysis)
5. [Target Distribution](#5-target-distribution)
6. [Numeric Feature Statistics](#6-numeric-feature-statistics)
7. [Text Feature Insights](#7-text-feature-insights)
8. [Correlation & Bivariate Insights](#8-correlation--bivariate-insights)
9. [Feature Engineering Summary](#9-feature-engineering-summary)
10. [Baseline Model Performance](#10-baseline-model-performance)
11. [Key Takeaways & Lessons](#11-key-takeaways--lessons)
12. [Strategy for Milestones 2–5](#12-strategy-for-milestones-25)

---

## 1. Dataset Overview

| File | Rows | Columns |
|------|------|---------|
| `train.csv` | **198,000** | 15 |
| `test.csv` | **102,000** | 14 (no `label`) |
| `Sample.csv` | **102,000** | 2 (id + label) |

> [!IMPORTANT]
> This is a **moderately large** dataset. With ~200k training samples we have plenty of data for complex models, but we must be careful about compute costs — especially for TF-IDF with high-dimensional text features.

The total ratio of test to train is roughly **1:2**, which is somewhat unusual (most competitions use 20-30% test). This suggests the hidden test set is substantial and the leaderboard score will be quite stable.

---

## 2. Missing Value Analysis

### Train Set

| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| `race` | 145,423 | **73.45%** |
| `religion` | 145,423 | **73.45%** |
| `gender` | 145,423 | **73.45%** |
| `comment` | 1 | 0.0005% |

### Test Set

| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| `race` | 75,269 | **73.79%** |
| `religion` | 75,269 | **73.79%** |
| `gender` | 75,269 | **73.79%** |

> [!IMPORTANT]
> **All three identity columns (`race`, `religion`, `gender`) are missing in ~73% of the data — and they are missing together.** This is NOT random — it's **MAR (Missing At Random)** or possibly **MNAR (Missing Not At Random)**. The missingness pattern is perfectly correlated across these three columns, meaning they either all have values or none of them do. This is a strong structural signal.

### What this means:
- These columns are only populated for ~27% of comments (52,577 out of 198,000)
- The platform likely only runs "identity detection" on certain comments (probably those flagged by some internal system)
- **The fact that these columns ARE populated is itself a feature** — it likely signals that the comment triggered some internal review
- We should create a binary feature `has_identity_signals` = 1 when race/religion/gender are NOT null

### The single missing `comment`:
- Only 1 comment is missing in train, 0 in test
- Decision: Fill with empty string or drop the row — negligible impact either way

---

## 3. Data Types — Critical Discoveries

```
Column        Dtype
------        -----
created_date  object (string, needs datetime parsing)
post_id       int64
emoticon_1    int64
emoticon_2    int64
emoticon_3    int64
upvote        int64
downvote      int64
if_1          int64
if_2          int64
race          object ← NOT numeric! Values like "none", NaN
religion      object ← NOT numeric! 8 unique values
gender        object ← NOT numeric! 5 unique values
disability    bool   ← True/False, not int!
comment       object
label         int64  (target: 0, 1, 2, 3)
```

> [!WARNING]
> **`race`, `religion`, `gender` are categorical strings, NOT numeric indicators!** The column description said "indicator" but the actual data contains values like `"none"`, plus other string categories. There are 6 unique race values, 8 religion values, and 5 gender values. We needed `pd.to_numeric(errors='coerce')` to convert them — but that means we lose the actual categorical information. A **better approach** for future milestones: use encoding (label/one-hot) for the non-null values.

> [!NOTE]
> **`disability` is boolean (True/False)**, not integer. Out of 198,000 rows, 195,257 are False (98.6%). Only ~1.4% of comments triggered the disability flag. This makes it an extremely imbalanced binary feature.

---

## 4. Duplicate Analysis

| Type | Count |
|------|-------|
| Exact duplicate rows | **0** |
| Rows with same `comment` text | **260** |

> [!NOTE]
> 260 rows share comment text with at least one other row. These are "partial duplicates" — same text but different metadata (different post_id, votes, timestamps). We **kept** them because they are likely legitimate reposts of the same comment in different threads.

---

## 5. Target Distribution

The target `label` has **4 classes** (0, 1, 2, 3). Based on `train.describe()`:
- mean: 0.794
- std: 0.980
- min: 0, max: 3
- 25th percentile: 0, 50th: 0, 75th: 2

### Key Observation:
The median is 0 and the 75th percentile is 2. This means **the majority of comments are label 0** (more than 50%), and the distribution is right-skewed. There is **class imbalance** — label 0 dominates.

> [!IMPORTANT]
> Since the competition metric is **Macro F1**, every class matters equally regardless of size. If we get 0.95 F1 on the majority class but 0.2 on the minority class, the macro average will be dragged down. This means:
> - We **must** use `stratify=y` in all train/val splits
> - We should experiment with `class_weight='balanced'` in models
> - SMOTE or other oversampling may help for minority classes
> - Per-class F1 analysis is essential to identify weak spots

---

## 6. Numeric Feature Statistics

| Feature | Mean | Std | Min | Q1 | Median | Q3 | Max | Key Observation |
|---------|------|-----|-----|-----------|--------|-----------|-----|-----------------|
| `post_id` | 68.4 | 27.9 | 20 | 39 | 72 | 72 | 129 | Categorical-like; dominated by post 72 |
| `emoticon_1` | 0.28 | 1.02 | 0 | 0 | 0 | 0 | 47 | Very sparse, mostly 0 |
| `emoticon_2` | 0.05 | 0.26 | 0 | 0 | 0 | 0 | 11 | Even sparser |
| `emoticon_3` | 0.12 | 0.48 | 0 | 0 | 0 | 0 | 17 | Sparse |
| `upvote` | 2.61 | 5.05 | 0 | 0 | 1 | 3 | 201 | Right-skewed, outliers |
| `downvote` | 0.67 | 2.04 | 0 | 0 | 0 | 1 | 107 | Mostly 0, heavy right tail |
| `if_1` | 1.91 | 25.6 | 0 | 0 | 0 | 4 | **1,860** | **EXTREME outliers** (std >> mean) |
| `if_2` | 7.96 | 14.8 | 3 | 4 | 6 | 10 | **1,833** | **EXTREME outliers**, min=3 (never 0) |

### Critical Insights:

1. **`if_1` and `if_2` have extreme outliers.** `if_1` max is 1,860 while mean is 1.9 — the std (25.6) is 13× the mean. This is a classic heavy-tailed distribution. `if_2` is similar with max 1,833 and std 14.8.
   - **For linear models:** These outliers will distort coefficients. Must use **RobustScaler** or **clip/winsorize** before feeding to LogReg/SVM.
   - **For tree-based models:** Trees are outlier-robust; no treatment needed.

2. **`post_id`** behaves like a low-cardinality categorical variable (only values between 20-129, with Q1=Q3=72). Frequency encoding is the right approach.

3. **Emoticon columns are extremely sparse** — most rows are 0. The sum of emoticons may be more useful than individual columns.

4. **`if_2` minimum is 3, not 0** — unlike every other numeric column. This suggests it's a different kind of internal system score, possibly always computed even for benign comments.

---

## 7. Text Feature Insights

### Comment Statistics:
From the sample output:
- `comment` has 197,999 non-null values (1 missing)
- 197,842 unique values — nearly every comment is unique
- The most common repeated comment is "Exactly.." (appears 8 times)

### What we built:
- `comment_len` / `comment_words` — length and word count distributions
- `comment_clean` — lowercased, URLs/mentions removed, digits → `NUM` token
- TF-IDF with 15,000 features, unigrams+bigrams, sublinear_tf
- `upper_ratio` — proportion of uppercase characters (signals shouting)
- `excl_count` / `ques_count` — punctuation-based emotion signals
- Per-label word clouds to visually confirm vocabulary differences

> [!TIP]
> **For Milestones 2-3:** Try increasing TF-IDF `max_features` to 20k-30k, or experiment with character-level n-grams alongside word-level. Also try different `min_df` and `max_df` thresholds.

---

## 8. Correlation & Bivariate Insights

### Key correlations (Spearman):
- `upvote` vs `downvote`: likely positively correlated (engaging comments get both)
- `if_1` vs `if_2`: their relationship reveals the internal scoring system
- `comment_len` vs `comment_words`: almost perfectly correlated (expected)
- Identity flags co-occur (all missing together or all present together)

### Kruskal-Wallis tests:
- If `comment_len` varies significantly across labels (p < 0.05), it confirms that comment length is informative for classification
- Emoticon presence may correlate with specific label categories

### Cramér's V (Cat × Cat associations):
- Binary indicators (race, religion, gender, disability) vs label associations reveal which identity signals are most predictive

---

## 9. Feature Engineering Summary

### Features created:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `comment_len_clean` | Character length of cleaned text | Toxicity often correlates with length |
| `comment_words_clean` | Word count of cleaned text | Similar signal to length |
| `total_votes` | upvote + downvote | Total engagement level |
| `vote_ratio` | upvote / (total_votes + 1) | Sentiment polarity of reactions |
| `vote_diff` | upvote – downvote | Net sentiment direction |
| `identity_flag_sum` | Sum of race+religion+gender+disability (after numeric conversion) | Intensity of identity mentions |
| `any_identity` | Binary: identity_flag_sum > 0 | Whether ANY identity group is mentioned |
| `emoticon_sum` | Sum of all emoticon indicators | Overall emoticon intensity |
| `upper_ratio` | % of uppercase chars in comment | Signals shouting/strong emotion |
| `excl_count` | Count of `!` in comment | Emotion intensity |
| `ques_count` | Count of `?` in comment | Questioning/sarcasm |
| `post_id_freq` | Frequency encoding of post_id | Controls for thread popularity |
| `hour`, `day_of_week`, `month`, `is_weekend` | Datetime extractions | Temporal posting patterns |

> [!TIP]
> **For next milestones:** Consider additional features:
> - Average word length (sophisticated vs simple language)
> - Special character ratio
> - Sentiment complexity (mixed positive/negative words)
> - Thread-level aggregates (average label of other comments in same post_id)
> - Interaction features: emoticon_sum × comment_length, etc.

---

## 10. Baseline Model Performance

### Model: Logistic Regression (multinomial, L2 penalty, C=1.0)

The baseline Logistic Regression was trained with:
- **15,000 TF-IDF features** (unigrams + bigrams, sublinear_tf)
- **~25 structured features** (votes, emoticons, identity flags, datetime, text stats)
- **StandardScaler** on structured features
- **80/20 stratified split** for validation

### Validation Results:
- **Macro F1 Score** on the validation set was computed
- **Classification report** showing per-class precision, recall, F1
- **Confusion matrix** visualised to identify which classes the model confuses

### 5-Fold Cross-Validation:
- Stratified 5-fold CV was run on the full training data
- Mean ± std of Macro F1 reported across folds

### Submission:
- Final model was retrained on ALL training data
- Predictions generated on test set
- Distribution of predictions checked for sanity
- Saved as `submission.csv`

> [!IMPORTANT]
> The baseline Logistic Regression score establishes the **floor** that all subsequent models must beat. This score is the benchmark for Milestones 2-5.

---

## 11. Key Takeaways & Lessons

### What we learned from the data:

1. **Text is the primary signal.** The `comment` column contains the most discriminative information. TF-IDF captures word-level patterns that separate categories.

2. **Identity columns are tricky.** They're mostly missing (~73%), storing string categories (not numeric), and their mere presence/absence is itself a feature.

3. **Extreme outliers exist.** `if_1` and `if_2` have values up to 1,860 — but tree-based models will handle these naturally.

4. **Class imbalance matters.** With Macro F1 as the metric, minority class performance is critical. We must monitor per-class F1 in every experiment.

5. **Feature engineering works.** Vote ratios, identity flag sums, uppercase ratios, and temporal features all add information beyond raw columns.

6. **Data leakage was prevented.** TF-IDF and scalers were fit on training split only, then applied to validation and test.

### What went wrong / needed fixing:

- **`race`, `religion`, `gender`** turned out to be string columns, not numeric indicators. We used `pd.to_numeric(errors='coerce')` to handle this, but this threw away the actual categorical values—future milestones should properly encode them.
- **`disability`** is boolean, not integer — needed explicit conversion.

---

## 12. Strategy for Milestones 2–5

### Milestone 2: Linear Models & SGD

**Goal:** Improve upon the LogReg baseline using SGD and hyperparameter tuning.

| Action Item | Description | Priority |
|-------------|-------------|----------|
| **SGD Classifier** | Train `SGDClassifier` with loss='log_loss' (equivalent to LogReg) and loss='hinge' (linear SVM). SGD scales better to large datasets. | HIGH |
| **Hyperparameter tuning** | GridSearch/RandomSearch over: `C` (regularization), `alpha` (for SGD), TF-IDF `max_features` (try 10k, 20k, 30k), `ngram_range` (try (1,3)), `class_weight` ('balanced' vs None) | HIGH |
| **Character n-grams** | Add character-level TF-IDF alongside word-level — captures subword patterns (e.g., "f***" variants) | MEDIUM |
| **Better identity encoding** | For the ~27% of data with identity values: Label-encode or one-hot encode `race`, `religion`, `gender` instead of using `pd.to_numeric` | HIGH |
| **Class weights** | Experiment with `class_weight='balanced'` — this adjusts for class imbalance automatically | HIGH |
| **Try different scalers** | Compare StandardScaler vs RobustScaler (better for outliers in `if_1`, `if_2`) | MEDIUM |

### Milestone 3: Dimensionality Reduction, Feature Selection & More Models

| Action Item | Description | Priority |
|-------------|-------------|----------|
| **PCA** | Apply PCA to TF-IDF features — try retaining 95%, 90%, 80% variance. Compare model performance with/without PCA. | MEDIUM |
| **Mutual Information** | Use `mutual_info_classif` to rank all features by their relationship with the target. Drop zero-information features. | HIGH |
| **RFECV** | Use Recursive Feature Elimination with CV to find optimal feature count for linear models | MEDIUM |
| **Naive Bayes** | `MultinomialNB` works very well with TF-IDF text features — often competitive baseline for text classification | HIGH |
| **KNN** | Try KNN on PCA-reduced features (full TF-IDF is too high-dimensional for KNN) | LOW |
| **SVM** | `SVC` with RBF kernel — requires scaled, reduced features. Try on PCA output. | MEDIUM |
| **Remove high-correlation features** | If `comment_len` and `comment_words` have r > 0.95, drop one | LOW |

### Milestone 4: Ensembles & MLP

| Action Item | Description | Priority |
|-------------|-------------|----------|
| **Random Forest** | Bagging ensemble — handles outliers naturally, good for mixed feature types | HIGH |
| **XGBoost/LightGBM** | Gradient boosting — typically best performance on tabular/mixed features. Try both. | CRITICAL |
| **Stacking** | Use outputs of LogReg, NB, RF, XGB as features for a meta-learner | HIGH |
| **Voting Ensemble** | Soft-vote combine top 3-4 models | MEDIUM |
| **MLP** | `MLPClassifier` on combined TF-IDF + structured features. Hidden layer sizes: (512, 256), (256, 128) | MEDIUM |
| **Blending** | Train models on different train folds, blend predictions on test | HIGH |

### Milestone 5: End-to-End Iteration

| Action Item | Description | Priority |
|-------------|-------------|----------|
| **Better text features** | Re-engineer text: try TruncatedSVD on TF-IDF, create topic-based features | HIGH |
| **Post-level features** | Aggregate label statistics within each `post_id` (mean label, std, count). Be careful of leakage — use only training data for these aggregates. | MEDIUM |
| **Feature interactions** | Create `emoticon_sum × comment_len`, `vote_ratio × identity_flag_sum`, etc. | MEDIUM |
| **Target encoding** | Target-encode `post_id` using leave-one-out strategy to prevent leakage | HIGH |
| **Threshold tuning** | After getting probability outputs, tune per-class prediction thresholds to maximize Macro F1 | CRITICAL |
| **Final submission** | Retrain best model on ALL training data. Verify prediction distribution matches training distribution. | CRITICAL |

---

## Quick Reference: Competition Constraints

| Constraint | Value |
|------------|-------|
| **Metric** | Macro F1 Score |
| **Target** | `label` (4 classes: 0, 1, 2, 3) |
| **Allowed libraries** | NumPy, Pandas, Matplotlib, Scikit-learn, XGBoost, Seaborn, Imblearn, SciPy, Pickle, regex, LightGBM, Plotly, statsmodels, wordcloud |
| **NOT allowed** | TensorFlow, PyTorch, NLTK, AutoML, spaCy, word2vec, textblob |
| **Final model** | Must retrain on full training data; scored on hidden Kaggle test set |

---

> [!CAUTION]
> **Critical reminder for all milestones:** Always fit preprocessors (TF-IDF, scalers, imputers, encoders) on training data ONLY. Transform test data using the parameters learned from training. Violating this rule causes data leakage and inflates validation scores.
