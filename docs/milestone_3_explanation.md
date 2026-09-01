# Milestone 3 — Cell-by-Cell Explanation & Viva Preparation

> This document explains **every design decision** in the Milestone 3 notebook.
> Use it to defend your methodology during the viva examination.

---

## Table of Contents
1. [What Changed from Milestone 2](#1-what-changed-from-milestone-2)
2. [Preprocessing — Why Identical to M2](#2-preprocessing--why-identical-to-m2)
3. [TruncatedSVD — Dimensionality Reduction](#3-truncatedsvd--dimensionality-reduction)
4. [Feature Selection — Chi-Squared & Mutual Information](#4-feature-selection--chi-squared--mutual-information)
5. [Naive Bayes — MultinomialNB & ComplementNB](#5-naive-bayes--multinomialnb--complementnb)
6. [K-Nearest Neighbours](#6-k-nearest-neighbours)
7. [Support Vector Machines — LinearSVC & Kernel SVM](#7-support-vector-machines--linearsvc--kernel-svm)
8. [Retraining M2 Models on Reduced Features](#8-retraining-m2-models-on-reduced-features)
9. [Calibrated SVC for Ensembling](#9-calibrated-svc-for-ensembling)
10. [Cross-Validation Strategy](#10-cross-validation-strategy)
11. [Final Submission Pipeline](#11-final-submission-pipeline)
12. [M2 Output Insights That Shaped M3](#12-m2-output-insights-that-shaped-m3)
13. [Viva Q&A — Likely Questions](#13-viva-qa--likely-questions)

---

## 1. What Changed from Milestone 2

| Problem in M2 | Fix/Extension in M3 | Why |
|---|---|---|
| Config D has 35,054 features → LogReg takes ~800s per fit | **Upgraded to 60k features** (35k word + 25k char) | More TF-IDF features = better F1 (proven in submitted notebooks) |
| Only tested linear models (LogReg, SGD, Ridge, PA) | Added **LinearSVC**, Naive Bayes, KNN | Covers SVM, probabilistic, and instance-based model families |
| No dimensionality reduction explored | TruncatedSVD (LSA) with variance analysis | Standard technique for sparse NLP data — showed it HURTS LinearSVC |
| No feature selection applied | Chi-Squared analysis | Showed nearly all TF-IDF features carry signal |
| LogReg `max_iter=1000` caused convergence warnings | Increased to `max_iter=1500` everywhere | M2 full run showed convergence issues |
| SMOTE applied to all models | **SMOTE removed entirely** | M2 showed: class_weight handles imbalance better |
| M2 total runtime: ~14+ hours | M3 completed in ~3-4 hours | Well within Kaggle's 12-hour limit |

### Actual M3 Results vs M2
| Metric | M2 Best | M3 Best | Improvement |
|---|---|---|---|
| Best Val F1 | 0.809 (PA) | **0.826 (LinearSVC)** | **+0.017** |
| Best CV F1 | 0.792 (PA) | **0.822 (LinearSVC)** | **+0.030** |
| Features | 35,054 | **60,054** | +71% |
| Best model | PA(C=0.01) | **LinearSVC(C=0.08-0.1)** | New champion |

---

## 2. Preprocessing — Why Identical to M2

**The preprocessing pipeline is 100% identical to Milestone 2.** This is a deliberate decision:

1. **Reproducibility:** Using the same feature engineering ensures that any performance differences are due to the *models*, not the features.

2. **No data leakage:** All transformers (TF-IDF, RobustScaler, SimpleImputer) are fit on the training split only and transformed on validation/test. This was established in M2 and preserved here.

3. **Upgraded TF-IDF:** Based on submitted notebook experiments (V5/V7), we increased TF-IDF from 35k to **60k features** (35k word + 25k char). This was proven to improve results significantly.

### Key preprocessing components:
- **Text cleaning:** lowercase, remove URLs/mentions, replace digits with `NUM`, collapse whitespace
- **Structured features (54):** engagement (votes, ratios), identity (one-hot + flags), text statistics (length, word count, uppercase ratio), datetime, post frequency
- **TF-IDF:** word-level (**35k**, trigrams, sublinear_tf) + char-level (**25k**, char_wb, 3-5 grams) = **60k total**
- **Scaling:** RobustScaler (median/IQR) — robust to extreme outliers in if_1, if_2

---

## 3. TruncatedSVD — Dimensionality Reduction

### Why TruncatedSVD (not PCA)?

| Technique | Input | Memory | Speed | Centering |
|---|---|---|---|---|
| PCA | Dense only | 198k × 60k × 8 bytes = **~93 GB** | Very slow | Centers data (destroys sparsity) |
| **TruncatedSVD** | **Sparse OK** | **Keeps sparse** | **Fast** | **No centering** |

PCA would require converting the sparse TF-IDF matrix to dense → impossible on Kaggle's 16GB RAM. TruncatedSVD operates directly on sparse matrices.

### Actual M3 Results: SVD Variance Explained
| Components | Cumulative Variance |
|---|---|
| 100 | 10.1% |
| 200 | 15.8% |
| 300 | 20.3% |
| 500 | 27.2% |

**Critical finding:** Even 500 components only capture 27.2% of the 60k-feature space. This means SVD loses too much information for LinearSVC — **use full sparse features instead.**

### What SVD Does (Mathematical Intuition)

TruncatedSVD decomposes the document-term matrix `X` as:

```
X ≈ U × Σ × V^T
```

Where:
- **U** (198k × n): document representations in the latent semantic space
- **Σ** (n × n): diagonal matrix of singular values (importance of each dimension)
- **V** (35k × n): term representations in the latent semantic space

By keeping only `n` components (instead of all 35k), we project documents into a lower-dimensional space that captures the most variance. This is called **Latent Semantic Analysis (LSA)**.

### Why L2 Normalisation After SVD?

SVD components have **very different magnitudes** — the first component captures much more variance than the 500th. Without normalisation, distance-based models (KNN, SVM) and even LogReg would be dominated by the high-variance components.

L2 normalisation puts every sample on the **unit sphere**, making all directions contribute equally. This is equivalent to making cosine similarity the implicit distance metric.

```python
normalizer = Normalizer(norm='l2')
X_tr_svd_norm = normalizer.fit_transform(X_tr_svd)
```

### How Many Components?

We used 200 components (15.8% variance retained). The choice is a **bias-variance trade-off**:

| n_components | Variance Explained | Effect |
|---|---|---|
| 100 | 10.1% | Far too little — massive information loss |
| **200** | **15.8%** | **Used for KNN** — only model that needs dense input |
| 500 | 27.2% | Still insufficient for competitive accuracy |

**Conclusion from M3 run:** SVD does NOT help LinearSVC. The full sparse 60k-feature space is where LinearSVC excels. SVD is only used to make KNN feasible.

### Why SVD Only on TF-IDF, Not Structured Features?

The 54 structured features are already low-dimensional and meaningful. SVD is designed to compress **high-dimensional, correlated** data — TF-IDF features are exactly this (many near-zero columns, correlated terms). Applying SVD to the 54 structured features would lose interpretability with no benefit.

```python
# SVD on TF-IDF only
X_tr_tfidf_only = hstack([X_tr_word, X_tr_char])  # 60,000 dims
X_tr_svd = svd.fit_transform(X_tr_tfidf_only)      # → 200 dims

# Then combine with structured (already dense, 54 dims)
X_tr_dense = np.hstack([X_tr_svd_norm, X_tr_struct])  # → 254 dims
```

---

## 4. Feature Selection — Chi-Squared

### Chi-Squared (χ²) Selection

**What it measures:** The statistical dependence between each feature and target label.

**How it works:** For each feature-label pair, compute:
```
χ² = Σ (observed - expected)² / expected
```

High χ² → feature strongly associated with the target → keep it.
Low χ² → feature is independent of target → remove it.

**Why it's perfect for TF-IDF:** χ² requires non-negative features. TF-IDF values are always ≥ 0. ✓

### Actual M3 Results: Chi² Feature Selection

| K (features kept) | LinearSVC F1 |
|---|---|
| 10,000 | 0.82126 |
| 20,000 | 0.82442 |
| 30,000 | 0.82463 |
| 40,000 | 0.82428 |
| 50,000 | **0.82563** |
| **All 60,054** | **0.82448** |

**Key finding:** Chi²(K=50k) marginally beat all features (0.82563 vs 0.82448), suggesting ~10k features are noise. But even dropping to K=10k only costs 0.003 F1 — most discriminative power is in the top features.

**Conclusion:** Feature selection provides negligible benefit and adds complexity. Keep all features.

### Feature Selection vs. Dimensionality Reduction

| Aspect | Feature Selection (Chi²) | Dimensionality Reduction (SVD) |
|---|---|---|
| **Approach** | Keeps original features, removes weak ones | Creates new features by combining existing ones |
| **Interpretability** | High — selected features are original terms | Low — components are abstract |
| **Feature space** | Subset of original | Entirely new space |
| **M3 Result** | Marginal benefit (±0.001) | **Hurts LinearSVC** |

---

## 5. Naive Bayes — MultinomialNB & ComplementNB

### Why Naive Bayes for Text?

Naive Bayes assumes **conditional independence** between features given the class:

```
P(class | features) ∝ P(class) × ∏ P(feature_i | class)
```

This assumption is clearly **wrong** for text (words are correlated: "not" and "good" are highly correlated). But it works surprisingly well because:
1. NB only needs to get the **ranking** of classes right, not their probabilities
2. The independence assumption **regularizes** the model — prevents overfitting
3. For bag-of-words/TF-IDF, each term's contribution is reasonably independent

### MultinomialNB vs ComplementNB

| | MultinomialNB | ComplementNB |
|---|---|---|
| **Models** | P(features \| class) | P(features \| NOT class) |
| **Bias** | Toward majority classes | **Corrected** — uses complement to reduce bias |
| **Best for** | Balanced datasets | **Imbalanced datasets like ours** |
| **Reference** | Standard NB | Rennie et al. (2003) |

**Why ComplementNB is critical here:** Our dataset has a 20.9:1 imbalance ratio (Label 0 vs Label 3). MultinomialNB's estimate of P(features|Label 3) is based on only 5,469 samples — noisy and unreliable. ComplementNB instead models P(features|NOT Label 3) using the 192,531 remaining samples — much more robust.

### Why NB on Sparse (Not SVD)?

NB requires **non-negative features**. TF-IDF is non-negative ✓. SVD output can be negative ✗.

For the structured features (which can be negative after RobustScaler), we use **MinMaxScaler** to rescale to [0, 1] before combining with TF-IDF:

```python
X_tr_tfidf_struct = hstack([X_tr_tfidf_only, csr_matrix(
    MinMaxScaler().fit_transform(imputer.transform(X_tr[structured_features]))
)])
```

### Hyperparameter: alpha (Smoothing)

The `alpha` parameter controls **Laplace smoothing**:
- `alpha=0` → no smoothing, zero counts cause zero probabilities (division by zero)
- `alpha=1` → add 1 to all counts (standard Laplace smoothing)
- Smaller alpha (0.01-0.1) → less smoothing, more trust in observed data

### Actual M3 Results: NB Performance
| Model | Alpha | Macro F1 |
|---|---|---|
| MultinomialNB | 0.1 | **0.573** |
| MultinomialNB | 0.5 | 0.555 |
| MultinomialNB | 1.0 | 0.511 |
| ComplementNB | 0.1 | 0.538 |
| ComplementNB | 0.5 | 0.536 |
| ComplementNB | 1.0 | 0.534 |

**Both NB variants are ~25 F1 points below LinearSVC (0.825).** The independence assumption is too restrictive for this dataset. Surprisingly, ComplementNB was WORSE than MultinomialNB — the 60k feature space may violate its assumptions more severely.

---

## 6. K-Nearest Neighbours

### Why KNN Needs Dimensionality Reduction

KNN computes distances between **all pairs of points**. With 35k features:

1. **Curse of dimensionality:** In high-D space, all distances become approximately equal. The nearest and farthest neighbours are almost equidistant. KNN loses discriminative power.

2. **Runtime:** Prediction is O(n × d × k). With n=158k, d=35k, each prediction takes ~5.5 billion operations. Impractical.

After SVD to 500-1000 dims: distances become meaningful again, and runtime drops by 35-70×.

### Distance Metrics

We test three metrics (all on SVD-reduced features):

| Metric | Formula | Best for |
|---|---|---|
| **Cosine** | `1 - cos(a,b)` | **Text/NLP** — measures angle, not magnitude |
| Euclidean | `‖a-b‖₂` | General purpose, scale-sensitive |
| Manhattan | `‖a-b‖₁` | High-dim, robust to outliers |

**Cosine distance is expected to win** for text data because TF-IDF vectors have very different magnitudes (some documents are much longer), but similar directions indicate similar content.

### weights='distance'

With `weights='distance'`, closer neighbours get **more weight** in the vote:
```
weight_i = 1 / distance(query, neighbour_i)
```

This is better than `weights='uniform'` (equal vote) because a very close neighbour should have more influence than a distant one.

### k values tested: 3, 5, 7

### Actual M3 Results: KNN Performance
| k | Metric | Macro F1 |
|---|---|---|
| 3 | cosine | 0.585 |
| 5 | cosine | 0.598 |
| **7** | **cosine** | **0.599** |

**KNN is the worst model tested** (0.60 vs 0.82 for LinearSVC). Even with SVD reduction to 200 dims, KNN cannot handle the class imbalance (no `class_weight` support) and the curse of dimensionality.

---

## 7. Support Vector Machines — LinearSVC & Kernel SVM

### Why SVM for Text Classification?

Text data is:
- **High-dimensional** (35k features)
- **Sparse** (most features are zero for any document)
- **Linearly separable** in high dimensions (each class has distinctive words)

These properties are **exactly** where SVMs excel. The maximum-margin hyperplane provides good generalisation, and the kernel trick is unnecessary for linear problems.

### LinearSVC vs SVC(kernel='linear')

| | LinearSVC | SVC(kernel='linear') |
|---|---|---|
| **Library** | `liblinear` | `libsvm` |
| **Complexity** | O(n × d) | O(n² × d) |
| **Time (Config D)** | **~15-30s** | **~hours** |
| **Probabilities** | ❌ (needs calibration) | ✅ with `probability=True` |

We use **LinearSVC** because `liblinear` is designed for large-scale sparse problems.

### LinearSVC on Sparse vs SVD-Reduced

We test LinearSVC in **two settings**:

1. **Full sparse (35k features):** Tests whether SVM can handle the full feature space — expected to be competitive with LogReg
2. **SVD-reduced (500-1000 features):** Tests whether dimensionality reduction helps or hurts SVM

### Kernel SVM (RBF) — Why Subsample?

RBF kernel SVM has **O(n²) training complexity** (must compute kernel matrix between all pairs). With n=158k:
```
n² = 158,000² = 24.96 billion entries
```

This is impractical. We **subsample to 30k samples** to make it feasible (~10-15 min). The trade-off: less training data, but we get to test a non-linear kernel.

### class_weight='balanced' for SVM

Same principle as LogReg — without balancing, SVM will always predict the majority class. With `class_weight='balanced'`:
```
weight_i = n_total / (n_classes × n_samples_class_i)
```

This makes misclassifying a Label 3 sample (2.76% of data) ~20× more costly than misclassifying a Label 0 sample (57.66%).

### C Parameter for SVM

`C` controls the **regularisation-margin trade-off**:
- Small C (0.01): wide margin, more misclassifications allowed → strong regularisation
- Large C (1.0-10.0): narrow margin, fewer misclassifications → fits training data closely

### Actual M3 Results: LinearSVC C-Value Sweep
| C | Val Macro F1 | Train Time |
|---|---|---|
| 0.01 | 0.812 | 359s |
| 0.05 | **0.826** | 539s |
| **0.08** | **0.826** | 649s |
| 0.1 | 0.824 | 670s |
| 0.15 | 0.823 | 765s |
| 0.5 | 0.816 | 913s |

**GridSearchCV (3-fold) confirmed C=0.1** as optimal (CV F1 = 0.822). The sweet spot is C=0.05-0.1.

**Note:** Unlike PA where C=0.01 was best, LinearSVC needs moderate C (0.08-0.1). This is because SVM's C directly trades off margin width vs training error, while PA's C controls the aggressiveness of weight updates — fundamentally different.

---

## 8. Retraining M2 Models on 60k Features

### Purpose

We retrained M2's best models on the new 60k feature config to establish baselines for M3's new models.

### Actual M3 Results: M2 Models on 60k Features

| Model | M2 (35k) F1 | M3 (60k) F1 | Change | Time (60k) |
|---|---|---|---|---|
| PA(C=0.01, balanced) | 0.809 | 0.793 | **−0.016** | 29s |
| LogReg(C=2.0, balanced) | 0.802 | **0.807** | +0.005 | 1369s |
| SGD(log_loss, balanced) | 0.787 | 0.793 | +0.006 | 16s |

**Key insights:**
- **PA got WORSE** with more features — PA's update rule is sensitive to feature dimensionality
- **LogReg improved slightly** but is extremely slow (1369s = 23 min per fit)
- **LinearSVC crushes all M2 models:** 0.826 vs best M2 model 0.807 (+0.019)

---

## 9. Calibrated SVC for Ensembling

### Why Calibrate LinearSVC?

LinearSVC's `predict()` returns class labels, but not probabilities. For **ensembling** (Milestone 4), we need probability estimates to combine classifiers.

`CalibratedClassifierCV` wraps LinearSVC with **Platt scaling** (sigmoid calibration):
1. Trains LinearSVC on each CV fold
2. Uses the held-out fold to fit a sigmoid function: `P(class) = 1 / (1 + exp(Af(x) + B))`
3. Result: calibrated probability estimates

```python
cal_svc = CalibratedClassifierCV(
    LinearSVC(C=best_C, class_weight='balanced', max_iter=2000),
    cv=3, method='sigmoid'
)
```

### method='sigmoid' vs method='isotonic'

- **sigmoid:** Parametric (2 parameters). Works well for binary/multi-class. Fast. Recommended for SVM.
- **isotonic:** Non-parametric. More flexible but overfit-prone with small calibration sets.

---

## 10. Cross-Validation Strategy

### Why 5-Fold Stratified CV?

1. **5-fold** (not 3 or 10): 5 is the standard balance between bias (too few folds) and variance (too many folds) + runtime
2. **Stratified:** Preserves class proportions in each fold — critical for our imbalanced data

### Actual M3 CV Results

| Model | CV Mean F1 | CV Std | Per-Fold Scores |
|---|---|---|---|
| **LinearSVC(C=0.1)** | **0.82233** | ±0.00387 | [0.815, 0.826, 0.824, 0.825, 0.823] |
| PA(C=0.01) | 0.81330 | ±0.00385 | [0.809, 0.814, 0.817, 0.818, 0.809] |

**LinearSVC wins CV** by 0.009 F1, with similar stability (same std). This confirms LinearSVC as the final submission model.

### Milestone Progression (CV scores)
| Milestone | Best CV Model | CV F1 |
|---|---|---|
| M2 | PA(C=0.01) | 0.792 |
| **M3** | **LinearSVC(C=0.1)** | **0.822** |

---

## 11. Final Submission Pipeline

### How the Final Model is Selected

1. Pick the **best CV model** from the 3 candidates
2. Retrain on **all training data** (198k samples, no validation holdout)
3. Transform test data using the **same** TF-IDF vectorizers and scalers

### Why Retrain on Full Data?

During experiments, we used 80/20 split → only 158k training samples. For the final submission, we use all 198k → **25% more training data**. This generally improves performance because:
1. More data → better generalisation
2. No held-out set needed since we already validated via CV

### Submission Distribution Check

Comparing prediction distribution to training distribution catches errors:
- If the model predicts 99% Label 0, class_weight likely wasn't applied
- If distributions are roughly proportional (with slight shift toward minority classes), the model is working correctly

---

## 12. M2 Output Insights That Shaped M3

### Key Findings from M2 Full Run

1. **PA(C=0.01) is the surprise performer** — fastest (15s/fit) AND highest single-split F1 (0.809). Very low C acts as strong regularisation, ideal for high-dim sparse data.

2. **LogReg(C=2.0, balanced) wins CV** — more robust across different train/val splits. Selected for final M2 submission.

3. **SMOTE has mixed results:**
   - Helped LogReg slightly (0.804 vs 0.802)
   - **Hurt SGD** (0.774 vs 0.788) — SMOTE on sparse TF-IDF creates dense synthetic samples that don't represent real text patterns

4. **Convergence warnings** — LogReg with `max_iter=1000` didn't converge on Config D. Fixed to `max_iter=1500` in M3.

5. **Config D consistently best** — char n-grams (+0.003-0.008 F1) and trigrams (+0.002-0.005 F1) both contributed. Total Config A→D improvement: ~0.01 F1.

6. **Runtime bottleneck** — Total M2 runtime was ~14+ hours, primarily due to LogReg (800s/fit) and GridSearchCV. M3 uses SVD reduction to make all models practical.

---

## 13. Viva Q&A — Likely Questions

### Q: Why TruncatedSVD instead of PCA?
**A:** PCA requires centering the data (subtracting the mean), which converts our sparse matrix to dense. A dense 198k × 35k float64 matrix needs ~52 GB RAM — impossible. TruncatedSVD works directly on sparse matrices. Mathematically, it's equivalent to PCA without centering.

### Q: What is Latent Semantic Analysis (LSA)?
**A:** LSA = TruncatedSVD applied to TF-IDF. It discovers latent topics — words that co-occur in similar documents map to the same SVD component. For example, "football", "goal", "team" might all load heavily on the same component, representing a "sports" topic.

### Q: Why not use PCA on the SVD output?
**A:** After SVD, the output is already dense and low-dimensional. PCA would just re-rotate the axes — no significant benefit. We use L2 normalisation instead.

### Q: How does ComplementNB handle class imbalance?
**A:** Standard MultinomialNB estimates P(word|class) → with only 5,469 Label 3 samples, these estimates are noisy. ComplementNB instead estimates P(word|NOT class) using the much larger complement set (192,531 samples) → much more stable. It then predicts the class whose complement model assigns the lowest probability.

### Q: Why doesn't KNN work well for text?
**A:** The "curse of dimensionality" — in 35k-dimensional space, all points are approximately equidistant. The ratio (max_distance - min_distance)/min_distance → 0 as dimensions grow. KNN's assumption that "nearby points have similar labels" breaks down. SVD reduction helps by projecting to a lower-dimensional space where distances are meaningful.

### Q: Why subsample for RBF kernel SVM?
**A:** RBF SVM needs to compute the kernel matrix K where K[i,j] = exp(-γ‖xᵢ-xⱼ‖²). This is n² entries. With n=158k: 25 billion entries × 8 bytes = 200 GB. Impossible. With n=30k: 900 million entries = ~7 GB. Feasible.

### Q: What's the difference between feature selection and dimensionality reduction?
**A:** Feature selection is a **filter** — keeps original features, removes unimportant ones. You can still interpret the remaining features ("word 'hate' is important"). Dimensionality reduction is a **transformation** — creates new features by combining existing ones. SVD components are abstract and not directly interpretable.

### Q: Why do you use class_weight='balanced' everywhere?
**A:** Our data has a 20.9:1 imbalance ratio (Label 0: 57.66% vs Label 3: 2.76%). Without balancing, models optimise for accuracy by predicting the majority class. With Macro F1 as our metric, all classes count equally — class_weight='balanced' penalises minority class errors more heavily, forcing the model to learn all classes.

### Q: What happens to performance when you reduce 60k features via SVD?
**A:** In M3, SVD(200) captured only 15.8% variance. For KNN, this was the only option (KNN needs dense input). But for LinearSVC, using full sparse features (0.825) is far better than SVD-reduced features. The high-dimensional sparse space is exactly where linear models excel — SVD discards discriminative features.

### Q: Why did Chi² at K=50k slightly beat using all features?
**A:** With 60k features, ~10k may be pure noise (very rare terms appearing in very few documents). Chi² removes these noisy features. But the difference is only 0.001 F1 — not worth the added pipeline complexity.

### Q: Why is LinearSVC faster than LogReg on the same features?
**A:** LinearSVC uses `liblinear` (coordinate descent optimiser) while LogReg with `lbfgs` uses limited-memory BFGS (quasi-Newton method). For sparse, high-dimensional problems, coordinate descent updates one feature at a time and skips zeros — very efficient. LBFGS maintains an approximation of the Hessian matrix — more memory-intensive and slower per iteration.

### Q: Why use `max_iter=1500` for LogReg?
**A:** M2's full run showed convergence warnings at `max_iter=1000` — the LBFGS optimizer hadn't converged. This means the model hadn't found the optimal weights, resulting in suboptimal predictions. `max_iter=1500` gives the optimizer enough iterations to converge on our 35k-feature dataset.

### Q: How do you prevent data leakage in the final submission?
**A:** The final submission pipeline **refits** all transformers (TF-IDF, SVD, scalers) on the full training data before transforming the test set. No information from the test set ever leaks into the training process. The sequence is:
1. `tfidf_word.fit_transform(all_train)` → `tfidf_word.transform(test)`
2. `scaler.fit_transform(all_train_struct)` → `scaler.transform(test_struct)`
3. `model.fit(all_train_features, all_labels)` → `model.predict(test_features)`
