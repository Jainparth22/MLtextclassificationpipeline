# Milestone 3 — Full Run Output Insights & Conclusion

> Analysis of the **28-milestone-3 output.ipynb** notebook.

---

## Pipeline Summary

| Item | Value |
|---|---|
| **Train shape** | (198,000 × 15) |
| **Test shape** | (102,000 × 14) |
| **Structured features** | 54 |
| **Train/Val split** | 158,400 / 39,600 (80/20 stratified) |
| **Word TF-IDF** | 35,000 (trigrams) |
| **Char TF-IDF** | 25,000 (char_wb, 3-5 grams) |
| **Total feature dims** | **60,054** |
| **Best model** | LinearSVC(C=0.1, balanced) |
| **Best Val F1** | **0.82554** (C=0.08 sweep) / **0.82448** (C=0.1 GridSearchCV) |
| **Best 5-Fold CV F1** | **0.82233 ± 0.00387** |

---

## Label Distribution (Same as M2)
| Label | Count | Proportion |
|---|---|---|
| 0 | 114,173 | 57.66% |
| 1 | 15,918 | 8.04% |
| 2 | 62,440 | 31.54% |
| 3 | 5,469 | 2.76% |

---

## Feature Configuration

Upgraded from M2's Config D (35,054 features) to a larger feature set:

| Component | M2 Config D | M3 Config | Change |
|---|---|---|---|
| Word TF-IDF | 25,000 | **35,000** | +10,000 |
| Char TF-IDF | 10,000 | **25,000** | +15,000 |
| Structured | 54 | 54 | Same |
| **Total** | **35,054** | **60,054** | **+25,000 (+71%)** |

---

## LinearSVC — The Clear Winner

### C-Value Sweep Results

| C value | Val Macro F1 | Train Time (s) |
|---|---|---|
| 0.01 | 0.81204 | 358.58 |
| 0.05 | **0.82552** | 539.11 |
| **0.08** | **0.82554** | 649.19 |
| 0.1 | 0.82448 | 670.26 |
| 0.15 | 0.82300 | 764.58 |
| 0.5 | 0.81580 | 912.57 |

**Key insight:** C=0.05 to C=0.1 is the sweet spot. C=0.08 gave the single highest F1, but the difference from C=0.1 is negligible (0.001).

### GridSearchCV (3-fold) Confirmed C=0.1

| Metric | Value |
|---|---|
| Best CV F1 | 0.82223 |
| Best params | C=0.1, class_weight='balanced' |
| Val F1 | 0.82448 |

### Best LinearSVC Per-Class Results

```
              precision    recall  f1-score   support
           0       0.98      0.94      0.96     22835
           1       0.74      0.84      0.78      3183
           2       0.88      0.90      0.89     12488
           3       0.61      0.74      0.67      1094

    accuracy                           0.91     39600
   macro avg       0.80      0.85      0.82     39600
weighted avg       0.92      0.91      0.91     39600
```

**Per-class observations:**
- **Label 0 (57.6%):** Excellent — 0.96 F1, very high precision (0.98)
- **Label 1 (8.0%):** Good — 0.78 F1, recall (0.84) better than precision (0.74)
- **Label 2 (31.5%):** Very good — 0.89 F1, balanced precision and recall
- **Label 3 (2.8%):** Decent — 0.67 F1, huge improvement from M2. Recall (0.74) much better than precision (0.61) due to class_weight='balanced' pushing the model to find minority samples even at the cost of false positives

---

## Dimensionality Reduction — TruncatedSVD Analysis

### Variance Explained

| Components | Cumulative Variance |
|---|---|
| 100 | 10.1% |
| 200 | 15.8% |
| 300 | 20.3% |
| 500 | 27.2% |

**Critical insight:** Even 500 SVD components only capture 27.2% of variance in the 60k-feature TF-IDF matrix. This is MUCH lower than the M3 submitted notebook (which had 91.1% at 200 components on a smaller 25k+25k config). The 60k feature space is extremely high-dimensional and not well-captured by SVD.

**SVD-reduced shape:** (158,400 × 254) — 200 SVD + 54 structured

---

## Chi-Squared Feature Selection Analysis

| Selection | K | LinearSVC F1 |
|---|---|---|
| Chi² | 10,000 | 0.82126 |
| Chi² | 20,000 | 0.82442 |
| Chi² | 30,000 | 0.82463 |
| Chi² | 40,000 | 0.82428 |
| Chi² | 50,000 | **0.82563** |
| **All features** | **60,054** | **0.82448** |

**Surprising finding:** Chi²(K=50k) actually **slightly beat** using all 60k features (0.82563 vs 0.82448). This suggests ~10k features may be noise or slightly harmful. However, the difference is marginal (0.001).

**Also notable:** Even dropping to K=10k only reduces F1 from 0.825 to 0.821 — a very small loss. This means most discriminative power is concentrated in the top 10k features, but the remaining features still help marginally.

---

## Naive Bayes Results (Poor Performance)

| Model | Alpha | Macro F1 | Train Time |
|---|---|---|---|
| MultinomialNB | 0.1 | **0.573** | 0.51s |
| MultinomialNB | 0.5 | 0.555 | 0.51s |
| MultinomialNB | 1.0 | 0.511 | 0.51s |
| ComplementNB | 0.1 | 0.538 | 0.50s |
| ComplementNB | 0.5 | 0.536 | 0.51s |
| ComplementNB | 1.0 | 0.534 | 0.51s |

**Conclusions:**
- **MultinomialNB(α=0.1) is the best NB** at 0.573, but still ~25 points below LinearSVC
- **Smaller alpha is better** for MultinomialNB (0.573 → 0.511 as alpha increases)
- **ComplementNB is WORSE than MultinomialNB** here — surprising given our class imbalance. The 60k feature space may violate ComplementNB's assumptions more severely.
- **NB is ~500× faster** than LinearSVC (0.5s vs 650s) but the performance gap is massive

---

## KNN Results (Very Poor Performance)

| k | Metric | Macro F1 | Train Time |
|---|---|---|---|
| 3 | cosine | 0.585 | 0.28s |
| 5 | cosine | 0.598 | 0.04s |
| **7** | **cosine** | **0.599** | 0.04s |

**Conclusions:**
- **KNN is the worst model family** tested (0.60 vs 0.82 for LinearSVC)
- Increasing k gives marginal improvement (k=7 best at 0.599)
- Even with SVD reduction to 200 dims, KNN cannot handle:
  - The class imbalance (no class_weight support)
  - The curse of dimensionality (200 dims is still too many for KNN)

---

## M2 Best Models on 60k Features

| Model | Config D (35k) F1 | 60k Features F1 | Change | Train Time (60k) |
|---|---|---|---|---|
| PA(C=0.01, balanced) | 0.809 | 0.793 | **−0.016** | 29.32s |
| LogReg(C=2.0, balanced) | 0.802 | **0.807** | **+0.005** | 1369.20s |
| SGD(log_loss, balanced) | 0.787 | 0.793 | **+0.006** | 15.75s |

**Key insights:**
- **PA got WORSE** with 60k features (0.809 → 0.793). PA's update rule is more sensitive to feature dimensionality — more features = more noise in the gradient.
- **LogReg improved slightly** (0.802 → 0.807) but is extremely slow (1369s = 22.8 min)
- **SGD improved slightly** (0.787 → 0.793) and remains very fast (15.75s)
- **LinearSVC crushes all M2 models:** 0.825 vs best M2 model 0.807

---

## Cross-Validation Results (Full Training Data)

| Model | CV Mean F1 | CV Std | Per-Fold Scores |
|---|---|---|---|
| **LinearSVC(C=0.1)** | **0.82233** | ±0.00387 | [0.815, 0.826, 0.824, 0.825, 0.823] |
| PA(C=0.01) | 0.81330 | ±0.00385 | [0.809, 0.814, 0.817, 0.818, 0.809] |

**LinearSVC wins CV** convincingly — 0.009 F1 higher than PA, with similar stability (similar std). Selected as the final submission model.

**Comparison with M2 CV results:**
| Model | M2 CV (35k) | M3 CV (60k) | Improvement |
|---|---|---|---|
| PA(C=0.01) | 0.79241 | 0.81330 | **+0.021** |
| LinearSVC | N/A | **0.82233** | New champion |

---

## Final Submission

| Property | Value |
|---|---|
| Model | LinearSVC(C=0.1, balanced) |
| Trained on | All 198,000 samples |
| Total features | 60,054 |

### Prediction Distribution

| Label | Count | Proportion | vs Training |
|---|---|---|---|
| 0 | 56,502 | 55.4% | 57.7% (lower) |
| 1 | 9,047 | 8.9% | 8.0% (higher) |
| 2 | 33,005 | 32.4% | 31.5% (similar) |
| 3 | 3,446 | 3.4% | 2.8% (higher) |

**Prediction vs training distribution is reasonable.** The model predicts slightly more Label 1 and Label 3 than exist in training — this is the expected effect of `class_weight='balanced'` pushing the model to find minority samples.

---

## Complete Model Rankings (All M3 Experiments)

| Rank | Model | Config | Macro F1 | Time (s) |
|---|---|---|---|---|
| 1 | **LinearSVC(C=0.08)** | Full Sparse (60k) | **0.82554** | 649 |
| 2 | LinearSVC(C=0.05) | Full Sparse (60k) | 0.82552 | 539 |
| 3 | LinearSVC(C=0.1) | Full Sparse (60k) | 0.82448 | 670 |
| 4 | LinearSVC(C=0.1, tuned) | Full Sparse (60k) | 0.82448 | — |
| 5 | LinearSVC(C=0.15) | Full Sparse (60k) | 0.82300 | 765 |
| 6 | LinearSVC(C=0.5) | Full Sparse (60k) | 0.81580 | 913 |
| 7 | LinearSVC(C=0.01) | Full Sparse (60k) | 0.81204 | 359 |
| 8 | LogReg(C=2.0, balanced) | Full Sparse (60k) | 0.80677 | 1369 |
| 9 | SGD(log_loss, balanced) | Full Sparse (60k) | 0.79308 | 16 |
| 10 | PA(C=0.01, balanced) | Full Sparse (60k) | 0.79257 | 29 |
| 11 | KNN(k=7, cosine) | SVD(200) | 0.59935 | 0.04 |
| 12 | KNN(k=5, cosine) | SVD(200) | 0.59756 | 0.04 |
| 13 | KNN(k=3, cosine) | SVD(200) | 0.58545 | 0.28 |
| 14 | MultinomialNB(α=0.1) | Full Sparse (NB) | 0.57335 | 0.51 |
| 15 | MultinomialNB(α=0.5) | Full Sparse (NB) | 0.55495 | 0.51 |
| 16 | ComplementNB(α=0.1) | Full Sparse (NB) | 0.53820 | 0.50 |
| 17 | ComplementNB(α=0.5) | Full Sparse (NB) | 0.53620 | 0.51 |
| 18 | ComplementNB(α=1.0) | Full Sparse (NB) | 0.53402 | 0.51 |
| 19 | MultinomialNB(α=1.0) | Full Sparse (NB) | 0.51118 | 0.51 |

---

## Key Conclusions

### 1. LinearSVC is the Undisputed Champion
LinearSVC(C=0.08-0.1) dominates all other models by a wide margin:
- **+0.02 over LogReg** (0.825 vs 0.807)
- **+0.03 over PA** (0.825 vs 0.793)
- **+0.23 over KNN** (0.825 vs 0.599)
- **+0.25 over NB** (0.825 vs 0.573)

### 2. 60k Features > 35k Features
Increasing TF-IDF dims from 35k to 60k improved:
- LinearSVC: new model not in M2, but achieves 0.825
- PA CV: 0.792 → 0.813 (+0.021)
- LogReg: 0.802 → 0.807 (+0.005)

### 3. SVD Does NOT Help LinearSVC
SVD captures only 27.2% variance at 500 components on the 60k matrix. The full sparse space contains critical discriminative features that SVD discards. **Use full sparse features for LinearSVC.**

### 4. Chi² Feature Selection is Marginally Useful
Chi²(K=50k) gave 0.82563 vs all features 0.82448 — a tiny +0.001. Not worth the added complexity. However, it shows ~10k features could be trimmed without loss.

### 5. NB and KNN Are NOT Competitive
- NB (0.51-0.57): Independence assumption fails badly
- KNN (0.59): Curse of dimensionality + no class weight support
- Both are ~25+ F1 points below LinearSVC

### 6. class_weight='balanced' Continues to Be Essential
Label 3 F1 improved from near-zero (without balancing) to 0.67 with balanced weights.

### 7. Runtime is Under Control
Total M3 runtime: ~3-4 hours (Kaggle), well within the 12-hour limit. The most expensive operation was the C-value sweep (6 models × ~600s each).

---

## Milestone Progression Summary

| Milestone | Best Model | Best Val F1 | Best CV F1 | Features |
|---|---|---|---|---|
| M1 | LogReg(C=1.0) | ~0.780 | — | 15k |
| M2 | PA(C=0.01) | 0.809 | 0.792 | 35k |
| **M3** | **LinearSVC(C=0.08-0.1)** | **0.826** | **0.822** | **60k** |

**Total improvement M1→M3:** +0.046 Macro F1 (+5.9% relative)
