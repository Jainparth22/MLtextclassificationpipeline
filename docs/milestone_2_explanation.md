# Milestone 2 — Cell-by-Cell Explanation & Viva Preparation

> **Erratum:** this document describes `RobustScaler` for the structured features. The code and notebooks (M2 onward) actually use `log1p` + `StandardScaler`; RobustScaler was considered but never used. Treat the RobustScaler passages as design rationale, not what was run.

> This document explains **every design decision** in the Milestone 2 notebook.
> Use it to defend your methodology during the viva examination.

---

## Table of Contents
1. [What Changed from Milestone 1](#1-what-changed-from-milestone-1)
2. [Imports & Setup](#2-imports--setup)
3. [Improved Preprocessing](#3-improved-preprocessing)
4. [TF-IDF Strategy — 4 Configurations](#4-tf-idf-strategy--4-configurations)
5. [Scaling — Why RobustScaler](#5-scaling--why-robustscaler)
6. [Logistic Regression Experiments](#6-logistic-regression-experiments)
7. [SGD Classifier — Theory & Experiments](#7-sgd-classifier--theory--experiments)
8. [Ridge Classifier](#8-ridge-classifier)
9. [Passive Aggressive Classifier](#9-passive-aggressive-classifier)
10. [SMOTE — Class Imbalance Handling](#10-smote--class-imbalance-handling)
11. [Hyperparameter Tuning](#11-hyperparameter-tuning)
12. [Data Leakage Prevention](#12-data-leakage-prevention)
13. [Viva Q&A — Likely Questions](#13-viva-qa--likely-questions)

---

## 1. What Changed from Milestone 1

| Problem in M1 | Fix in M2 | Why |
|---|---|---|
| `race`, `religion`, `gender` converted with `pd.to_numeric` (lossy) | Proper one-hot encoding + `has_identity_data` binary | Preserves categorical information instead of converting "none"→NaN→0 |
| Only `StandardScaler` used | `RobustScaler` for structured features | Uses median/IQR instead of mean/std — robust to extreme outliers in `if_1`, `if_2` |
| Only word-level TF-IDF (15k) | 4 configs: word(15k), word(25k+trigrams), word+char, word(25k)+char | Character n-grams capture sub-word patterns (censored words, typos) |
| No SMOTE | SMOTE + `class_weight='balanced'` compared | Two complementary approaches to class imbalance |
| Single LogReg model | LogReg + SGD(3 losses) + Ridge + Passive-Aggressive | Comprehensive linear model comparison |
| Manual C tuning | GridSearchCV + RandomizedSearchCV | Systematic, cross-validated hyperparameter search |
| No log transforms | `log1p` on if_1, if_2, upvote, downvote | Reduces skewness, helps linear models |
| No avg_word_len / special_char_ratio | Added both | Vocabulary sophistication + special character density signals |

---

## 2. Imports & Setup

### New imports over M1:
```python
from sklearn.linear_model import SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
```

**Why `RobustScaler`?** StandardScaler uses mean and std, which are sensitive to outliers. `if_1` has mean=1.9 but max=1860 — the std (25.6) is 13× the mean. RobustScaler uses **median and interquartile range (IQR)** instead, which are robust statistics.

**Why `SMOTE`?** Synthetic Minority Over-sampling TEchnique creates new synthetic examples of minority classes by interpolating between existing ones. It works in **feature space**, not by duplicating rows.

**Why `CalibratedClassifierCV`?** Some models (Ridge, SGD with hinge loss) don't output proper probabilities. CalibratedClassifierCV wraps them to provide calibrated probability estimates using Platt scaling or isotonic regression.

---

## 3. Improved Preprocessing

### 3.1 Identity Column Encoding

**Problem:** In M1, we used `pd.to_numeric(train['race'], errors='coerce')`. This converts:
- `"none"` → NaN → 0
- `"white"` → NaN → 0
- `NaN` → NaN → 0

All three values become 0 — we lose all categorical information!

**Solution in M2:**
```python
# 1. Create binary: does this row have identity data at all?
df['has_identity_data'] = df[identity_str_cols].notna().any(axis=1).astype(int)

# 2. Fill NaN with 'missing', then one-hot encode
df[col] = df[col].fillna('missing')
train_identity_dummies = pd.get_dummies(train[identity_str_cols], drop_first=True)
```

**Why `drop_first=True`?** To avoid the "dummy variable trap" — if you have N categories, you only need N-1 binary columns. The dropped category is implicitly represented when all other columns are 0.

**Why `has_identity_data`?** The ~73% missing pattern is itself a strong signal. If the platform has identity data, it means the comment was flagged for review. This binary feature captures that structural information.

### 3.2 Log Transforms

```python
df['log_if_1'] = np.log1p(df['if_1'])
df['log_upvote'] = np.log1p(df['upvote'])
```

**Why `log1p` (not `log`)?** `log(0)` is undefined. `log1p(x) = log(1+x)` handles zeros safely. The result: `log1p(0)=0`, `log1p(1)=0.69`, `log1p(1860)=7.53`. This compresses the extreme tail dramatically, making the distribution more normal — which linear models prefer.

### 3.3 New Features

| Feature | Formula | What it captures |
|---|---|---|
| `avg_word_len` | mean word length in comment | Vocabulary sophistication (longer words = more educated writing) |
| `special_char_ratio` | count of non-alnum, non-space / length | Lots of special chars may indicate anger (***, !!!, ???) |
| `has_emoticon` | binary: any emoticon > 0 | Whether the comment uses any emoticons at all |
| `log_upvote/downvote` | log1p(votes) | Dampened vote counts for linear models |
| `log_if_1/if_2` | log1p(if values) | Dampened internal feature scores |

---

## 4. TF-IDF Strategy — 4 Configurations

### Why multiple TF-IDF configs?

Text is the **primary signal** for classification. The TF-IDF settings dramatically affect what the model sees. We don't know a priori which settings work best, so we test systematically.

| Config | max_features | ngram_range | analyzer | Total dims |
|---|---|---|---|---|
| **A** (M1 baseline) | 15,000 | (1,2) | word | ~15,000 + structured |
| **B** (larger) | 25,000 | (1,3) | word | ~25,000 + structured |
| **C** (word+char) | 15,000 + 10,000 | word:(1,2), char:(3,5) | both | ~25,000 + structured |
| **D** (full) | 25,000 + 10,000 | word:(1,3), char:(3,5) | both | ~35,000 + structured |

### Why character n-grams?

Character n-grams within word boundaries (`char_wb`) capture:
- **Censored profanity:** "f\*\*k", "sh\*t" share character patterns with the actual words
- **Misspellings:** "fascsim" (from the actual data!) matches "fascism" at the character level
- **Morphological patterns:** "-ing", "-tion", "-ism" prefixes/suffixes carry meaning

Example: `ngram_range=(3,5)` on word "fascist" produces:
`" fa"`, `"fas"`, `"asc"`, `"sci"`, `"cis"`, `"ist"`, `"st "`, `" fas"`, `"fasc"`, `"asci"`, `"scis"`, `"cist"`, `"ist "`, ...

### Why `sublinear_tf=True`?

Instead of raw term frequency, it uses `1 + log(tf)`. This prevents very frequent terms from dominating. Without it, a word appearing 100 times would have 100× the weight of a word appearing once. With sublinear, it's only `1+log(100) ≈ 5.6×`.

### Why `stop_words='english'`?

Removes common words like "the", "is", "at" that carry no discriminative power. This reduces dimensionality and noise. Only applied to word-level TF-IDF (character n-grams don't have stop words).

---

## 5. Scaling — Why RobustScaler

### StandardScaler vs RobustScaler:

| Aspect | StandardScaler | RobustScaler |
|---|---|---|
| **Center** | mean | **median** |
| **Scale** | std | **IQR (Q3-Q1)** |
| **Outlier sensitivity** | Very sensitive | **Robust** |
| **When to use** | Normal-ish data | Data with outliers |

Our data has `if_1` with mean=1.9 but max=1860. StandardScaler would scale most values close to 0 and the outlier to ~72 standard deviations — linear models will be dominated by these extreme values. RobustScaler uses the middle 50% of the data (IQR), so outliers don't distort the scaling.

### Why apply to structured features only?

TF-IDF output is already normalised (L2 norm by default). Re-scaling TF-IDF would distort the carefully computed term weights. We only scale the structured (numeric) features.

---

## 6. Logistic Regression Experiments

### 6.1 Why multinomial (not OVR)?

`multi_class='multinomial'` fits all 4 classes simultaneously using the softmax function. `ovr` (one-vs-rest) trains 4 separate binary classifiers. Multinomial is generally better when classes are mutually exclusive (which they are here — each comment has exactly one label).

### 6.2 Why `class_weight='balanced'`?

With imbalanced classes, the model optimises for accuracy — which means it learns to predict the majority class. `class_weight='balanced'` automatically adjusts the loss function:

```
weight_for_class_k = n_samples / (n_classes × n_samples_in_class_k)
```

For a class with 60% of data: weight = 198000 / (4 × 118800) ≈ 0.42
For a class with 10% of data: weight = 198000 / (4 × 19800) ≈ 2.5

This forces the model to "care" equally about all classes — which is exactly what Macro F1 rewards.

### 6.3 C-value sweep

`C` is the **inverse** of regularisation strength:
- **Small C (e.g., 0.01):** Strong regularization → simpler model, may underfit
- **Large C (e.g., 10.0):** Weak regularization → complex model, may overfit

We sweep: `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]` to find the sweet spot. Too small → underfits (poor F1 on all classes). Too large → overfits to training data noise.

### 6.4 Solver choice

- **`lbfgs`:** Default, good for multinomial + L2. Uses limited-memory BFGS quasi-Newton method.
- **`saga`:** Supports L1, L2, and Elastic Net. Better for very large datasets. Uses stochastic average gradient descent.

We test both in GridSearchCV.

---

## 7. SGD Classifier — Theory & Experiments

### Why SGD?

**Stochastic Gradient Descent** updates model weights one sample at a time (or mini-batch), instead of computing the gradient over the entire dataset. This makes it:
- **Much faster** for large datasets (our ~200k rows)
- **Memory efficient** — doesn't need to hold all data in memory
- **Equivalent** to LogReg/SVM when using the right loss function

### Loss functions:

| Loss | Equivalent to | Probability output? | Notes |
|---|---|---|---|
| `log_loss` | Logistic Regression | Yes | Smooth, differentiable, default choice |
| `hinge` | Linear SVM | No (needs calibration) | Maximum-margin classifier, sparse in dual |
| `modified_huber` | Smoothed SVM | Yes | Compromise: margin-based but gives probabilities |

### Alpha vs C:

SGD uses `alpha` (direct regularisation strength), while LogReg uses `C` (inverse). Rough equivalence: `alpha ≈ 1 / (n_samples × C)`. For 160k training samples and C=1: `alpha ≈ 1/160000 ≈ 6e-6`.

### Elastic Net penalty:

```python
penalty='elasticnet', l1_ratio=0.3
```

Elastic Net = `l1_ratio × L1 + (1-l1_ratio) × L2`
- **L1 (Lasso):** Drives unimportant feature weights to exactly zero → automatic feature selection
- **L2 (Ridge):** Shrinks all weights toward zero but never to exactly zero → prevents overfitting
- **Elastic Net:** Best of both worlds. L1 selects features, L2 handles correlated features

With 15,000+ TF-IDF features, many are probably noise. L1 component helps zero them out.

---

## 8. Ridge Classifier

### How it differs from Logistic Regression:

- LogReg minimises **cross-entropy loss** (probabilistic)
- Ridge minimises **squared error loss** (regression-style) on class indicators

Ridge converts the classification problem to regression: each class becomes a 0/1 target, and it fits a least-squares regression with L2 penalty. Prediction = class with highest predicted value.

### Why use it?

Ridge is often **much faster** than LogReg because it solves a linear system directly (no iterative optimisation). For text classification, it's surprisingly competitive.

---

## 9. Passive Aggressive Classifier

### How it works:

The PA algorithm is an **online learning** algorithm with a specific update rule:
- If a sample is classified correctly with sufficient margin → **no update** (passive)
- If a sample is misclassified → **update aggressively** with the minimum change needed to classify it correctly

### C parameter meaning:

Unlike LogReg where C is inverse regularization, in PA the `C` parameter controls the maximum step size of each update. Larger C → more aggressive corrections → faster learning but more sensitive to noise.

---

## 10. SMOTE — Class Imbalance Handling

### Theory:

SMOTE works in **feature space**:
1. Pick a minority class sample
2. Find its k nearest neighbours (within the same class)
3. Create a synthetic sample by interpolating between them

```
synthetic = sample + random(0,1) × (nearest_neighbour - sample)
```

### SMOTE vs class_weight:

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| `class_weight='balanced'` | Adjusts loss function weights | No synthetic data, fast | Doesn't add diversity |
| SMOTE | Creates synthetic minority samples | Adds diversity, better decision boundaries | Slower, may create noisy samples |

### Why we tested both:

SMOTE is NOT always better. For text data (especially sparse TF-IDF), interpolating between sparse vectors can create unrealistic samples. `class_weight` is often sufficient for text classifiers. We test both and compare.

> [!WARNING]
> **SMOTE must be applied ONLY to training data, NEVER to validation/test data.** Applying SMOTE before the train/val split would create synthetic samples that "leak" information from validation into training.

---

## 11. Hyperparameter Tuning

### GridSearchCV vs RandomizedSearchCV:

| Method | How it works | When to use |
|---|---|---|
| **GridSearchCV** | Tests ALL combinations in the parameter grid | Few parameters, small grids |
| **RandomizedSearchCV** | Randomly samples N combinations from distributions | Many parameters, continuous ranges |

### Why 3-fold CV (not 5-fold)?

With 160k training samples and potentially 40+ parameter combinations in RandomizedSearch, 5-fold CV would be very slow. 3-fold is a reasonable trade-off — each fold still has ~53k samples for training, which is plenty.

### Parameter distributions for SGD:

```python
'alpha': loguniform(1e-6, 1e-1)
```

Why `loguniform`? Because regularisation parameters span orders of magnitude. A uniform distribution would oversample the high end. `loguniform` samples equally across log scale: equal probability of getting 1e-5 or 1e-3.

---

## 12. Data Leakage Prevention

### What is data leakage?

Data leakage occurs when information from the test/validation set "leaks" into the training process. This inflates validation scores but hurts real-world performance.

### Where leakage can occur in our pipeline:

| Step | Risk | How we prevent it |
|---|---|---|
| TF-IDF fitting | If fit on all data, IDF weights include test set word frequencies | **Fit only on X_train split** |
| Scaling | If fit on all data, median/IQR include test set statistics | **Fit only on X_train split** |
| Imputation | If fit on all data, median includes test set values | **Fit only on X_train split** |
| SMOTE | If applied before split, synthetic samples may mirror validation data | **Apply only after split** |
| Post-ID frequency encoding | If computed on all data, frequencies include test set | **Computed on train only** |
| Identity encoding | If categories come from test set, leaks test distribution | **get_dummies aligned, never fit on test** |

### The golden rule:

> **Any transformation that "learns" from data (fit/fit_transform) must ONLY see training data.** Test/validation data is ONLY transformed, never fitted.

---

## 13. Viva Q&A — Likely Questions

### Q: "Why did you use RobustScaler instead of StandardScaler?"

**A:** "Our EDA in Milestone 1 revealed that `if_1` and `if_2` have extreme outliers — `if_1` has a mean of 1.9 but max of 1,860, and its standard deviation is 13 times its mean. StandardScaler uses mean and standard deviation, which are heavily distorted by outliers. RobustScaler uses the median and interquartile range — statistics that are inherently robust to outliers. This means 95% of the data gets properly scaled, while outliers are naturally compressed rather than dominating the feature space. For linear models like Logistic Regression, proper scaling is essential because the coefficients are directly proportional to feature magnitudes."

### Q: "Why test 4 different TF-IDF configurations?"

**A:** "Text is our primary predictive signal, and TF-IDF settings directly control what information the model sees. We tested four configurations:
1. **Baseline (15k words)** — establishes the M1 reference point.
2. **Larger vocabulary (25k + trigrams)** — captures longer phrases like 'climate change denier'.
3. **Word + character n-grams** — character-level captures sub-word patterns: censored profanity, misspellings (we saw 'fascsim' in our EDA), and morphological patterns.
4. **Full combination** — tests whether more features help or cause overfitting.

By comparing Macro F1 across all four, we can quantify the marginal value of each TF-IDF enhancement."

### Q: "Explain the difference between SGD with log_loss and hinge loss."

**A:** "Both are linear classifiers, but they optimise different objectives:
- **log_loss (Logistic Regression)** minimises cross-entropy — it tries to make the predicted probability of the correct class as high as possible. It always outputs well-calibrated probabilities.
- **hinge loss (Linear SVM)** maximises the margin between classes — it doesn't care about probability, only that the correct class score is at least 1 unit higher than any other class. This creates sparser solutions where only 'support vectors' near the decision boundary matter.

In practice, log_loss is smoother and easier to optimise, while hinge can give better generalisation for well-separated classes. We test both because our classes may have different separability in the TF-IDF feature space."

### Q: "Why is SMOTE applied AFTER the train-val split?"

**A:** "If we applied SMOTE before splitting, synthetic samples created from validation data would end up in the training set. The model would then be indirectly trained on validation data — this is data leakage. SMOTE must only see training examples, so synthetic samples are purely derived from the training distribution. The validation set remains untouched and representative of the true test distribution."

### Q: "What is Elastic Net and why did you try it?"

**A:** "Elastic Net combines L1 (Lasso) and L2 (Ridge) regularisation:

`L_elasticnet = α × [l1_ratio × ||w||₁ + (1-l1_ratio) × 0.5 × ||w||₂²]`

L1 drives unimportant feature weights to exactly zero — this is automatic feature selection. With 15,000+ TF-IDF features, many are noise. L1 zeros them out. But L1 alone struggles with correlated features (picks one arbitrarily). L2 handles correlated features gracefully by distributing weight among them. Elastic Net gets the best of both: feature selection from L1, stability from L2.

The `l1_ratio` parameter controls the balance: 0.0 = pure L2, 1.0 = pure L1. We swept values [0.1, 0.3, 0.5, 0.7, 0.9] to find the optimal mix."

### Q: "How does class_weight='balanced' help with Macro F1?"

**A:** "Macro F1 computes F1 for each class independently, then averages — giving equal importance to every class regardless of size. A model that gets 0.95 F1 on the majority class (60% of data) but 0.2 F1 on a minority class (10%) would have a poor Macro F1.

`class_weight='balanced'` adjusts the loss function so that misclassifying a minority sample costs more than misclassifying a majority sample. Specifically, the weight for class k is `n_samples / (n_classes × count_of_class_k)`. This means a minority class with 10% of data gets 2.5× the weight of a class with 25%. The model therefore optimises equally for all classes — directly aligned with Macro F1."

### Q: "Why did you log-transform if_1 and if_2?"

**A:** "Linear models assume a roughly linear relationship between features and the log-odds of the target. With if_1 ranging from 0 to 1,860 and most values clustered near 0, the original scale creates a highly skewed distribution. The log transform (`log1p` to handle zeros) compresses the range to [0, ~7.5], making the distribution more symmetric and the linear relationship more appropriate. This is a well-known technique in statistics called 'variance-stabilising transformation' — it makes the variance approximately constant across the range of the feature."

### Q: "What's the difference between GridSearchCV and RandomizedSearchCV?"

**A:** "GridSearchCV exhaustively evaluates ALL parameter combinations. If we have 5 values for C, 2 for solver, and 2 for class_weight — that's 5×2×2=20 combinations, each evaluated with 3-fold CV = 60 model fits. Manageable.

For SGD, we have continuous parameters like alpha (any value from 1e-6 to 0.1), l1_ratio (0 to 1), plus discrete choices for loss, penalty, class_weight. The full grid would be enormous. RandomizedSearchCV instead samples N random combinations from the parameter distributions. With N=40, we explore the space efficiently without being exhaustive. Research shows that random search is often better than grid search for the same compute budget, because it doesn't waste time on unimportant parameter axes."
