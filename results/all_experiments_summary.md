# Complete Model & Data Insights — All Submitted Notebooks

> Comprehensive summary of all experiments, results, and insights from every
> notebook in the "Now submitted Notebook" folder, plus M1 and M2 outputs.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Train samples | 198,000 (one row dropped → 197,999) |
| Test samples | 102,000 |
| Original features | 15 columns |
| Engineered structured features | 54 (after identity dummies, engagement, text stats, datetime) |

### Label Distribution (Severe Imbalance)
| Label | Count | Proportion | Role |
|---|---|---|---|
| 0 | 114,173 | 57.66% | Majority |
| 1 | 15,918 | 8.04% | Minority |
| 2 | 62,440 | 31.54% | Second largest |
| 3 | 5,469 | 2.76% | Extreme minority |

**Imbalance ratio:** 20.9:1 (Label 0 vs Label 3)

---

## 2. Feature Configurations Explored

### M2 Configs (word + char TF-IDF + structured)
| Config | Word TF-IDF | Char TF-IDF | Structured | Total Dims |
|---|---|---|---|---|
| A | 15k (bigrams) | — | 54 | 15,054 |
| B | 25k (trigrams) | — | 54 | 25,054 |
| C | 15k (bigrams) | 10k | 54 | 25,054 |
| **D** | **25k (trigrams)** | **10k** | **54** | **35,054** |

### V5/V7 Config (final submitted notebooks)
| Component | Setting | Dims |
|---|---|---|
| Word TF-IDF | **35k**, trigrams, no stop_words removal | 35,000 |
| Char TF-IDF | **25k**, char_wb, (3,5)-grams | 25,000 |
| Structured | 19 features (simplified from 54) | 19 |
| **Total** | | **60,019** |

**Key finding:** The final V5/V7 notebooks increased TF-IDF dimensions from 35k to **60k** (35k word + 25k char) and this significantly improved results. Also notably, V5 did NOT use stop_words removal and simplified text cleaning (minimal, no stemming).

---

## 3. All Model Results Across All Notebooks

### Milestone 1 — Baseline (Config A: 15k word TF-IDF)
| Model | Macro F1 | Notes |
|---|---|---|
| LogReg (C=1.0, no balancing) | ~0.780 | M1 baseline |

### Milestone 2 — Linear Models (Config D: 35k features)
| Rank | Model | Macro F1 | Train Time |
|---|---|---|---|
| 1 | PA(C=0.01) | **0.80896** | 14.62s |
| 2 | LogReg+SMOTE | 0.80441 | 1674.45s |
| 3 | LogReg(C=2.0, balanced) | 0.80216 | 813.83s |
| 4 | LogReg(balanced) | 0.80077 | 564.49s |
| 5 | SGD(hinge/SVM) | 0.79530 | 8.66s |
| 6 | SGD(RandomSearch) | 0.79763 | — |
| | PA 5-Fold CV Mean | 0.79241±0.015 | — |

### Milestone 3 — Submitted Notebook (Config: word 25k + char 25k)
| Model | Macro F1 | Train Time | Notes |
|---|---|---|---|
| **Tuned LinearSVC** | **0.8698** | — | C=0.1, balanced |
| Tuned PA | 0.8595 | — | C=0.01, balanced |
| LR (balanced, 25K) | 0.7979 | 218.5s | On 25k word only |
| PA (balanced, default) | 0.7019 | 34.1s | Before tuning |
| Tuned MultinomialNB | 0.6687 | — | alpha tuned |
| Tuned ComplementNB | 0.6300 | — | alpha=1.0 |
| Tuned KNN (SVD 200-dim) | 0.5615 | 695s | k=5, distance |
| KNN (k=5, SVD) | 0.5564 | 48.3s | Very poor |

### M5-V5 — Final Submitted (Config: 60k features = 35k word + 25k char + 19 structured)
| Model | Macro F1 | Train Time | Notes |
|---|---|---|---|
| **LinearSVC (custom weights)** | **0.8221** | 329s | C=0.1 |
| Calibrated LinearSVC | 0.8070 | — | For ensembles |
| Stacking (4→LR) | **0.8266** | — | Best ensemble |
| Voting Ensemble | ~0.82 | — | Soft voting |

### M5-V7 — Final Optimised (Config: 60k features, tuned C=0.08)
| Model | Macro F1 | Train Time | Notes |
|---|---|---|---|
| **LinearSVC (tuned C=0.08)** | **0.8227** | 235s | Best individual |
| Calibrated LinearSVC | 0.8052 | 479s | |
| LogReg | 0.8079 | 230s | |
| SGD | 0.8074 | 9.3s | |
| PA | 0.7638 | 20.6s | Much worse here |
| **Stacking (4→LR)** | **0.8267** | 1129s | Best overall |
| Seed-averaged (5 models) | ~0.826 | — | Final submission |

---

## 4. Key Insights & Conclusions

### A. Feature Engineering Insights

1. **More TF-IDF features = better:** Config D (35k) beat Config A (15k) by ~0.01 F1. V5's 60k features pushed further.

2. **Char n-grams are critical:** Consistently added +0.003-0.008 F1. They capture sub-word patterns (censored words like "f***", typos, morphological variations).

3. **Trigrams matter:** Word trigrams added +0.002-0.005 F1 over bigrams.

4. **Stop words removal is optional:** V5/V7 (no stop_words removal) achieved the highest scores. TF-IDF's sublinear_tf and max_df=0.95 already downweight common words.

5. **Structured features help but are secondary:** The 54 engineered features contribute ~0.002-0.005 F1 on top of TF-IDF.

### B. Model Insights

1. **LinearSVC is the clear winner** for this problem:
   - Best individual model across ALL notebooks (0.82-0.87 F1)
   - Much faster than LogReg on high-dim sparse data
   - `class_weight='balanced'` or custom weights essential

2. **Custom class weights > balanced:**
   ```python
   CUSTOM_WEIGHTS = {0: 1.0, 1: 5.5, 2: 1.8, 3: 14.0}
   ```
   This produced better results than sklearn's auto-balanced weights.

3. **PA(C=0.01) was M2's star but fell behind in later notebooks:**
   - M2: 0.809 (best) → M5-V7: 0.764 (much worse with 60k features)
   - PA benefits more from regularisation (small C) — with 60k features, it needs even smaller C

4. **Naive Bayes is disappointing:** Both MultinomialNB (~0.67) and ComplementNB (~0.63) are far below linear models. The independence assumption is too strong.

5. **KNN is terrible:** 0.56 F1 even with SVD reduction. The curse of dimensionality and lack of class_weight support makes it unusable.

6. **Stacking is the best ensemble:** 4 base models (CalibratedSVC, CalibratedLR, SGD, PA) → LR meta-learner achieves 0.827 F1.

7. **SMOTE/ROS are unnecessary:** V5 conclusion: "No ROS/SMOTE — class_weight handles imbalance better." This was confirmed across multiple notebooks.

### C. Dimensionality Reduction Insights

1. **TruncatedSVD 200 components = 91.1% variance** on the M3 submitted config.

2. **SVD hurts LinearSVC dramatically:** LinearSVC on full sparse features (0.87) vs SVD-reduced (much lower). SVD removes discriminative features.

3. **Chi² feature selection at K=60k (all features) = best.** Reducing K dropped F1 significantly (0.82 → 0.71). This means nearly all TF-IDF features carry signal.

4. **Conclusion: Don't reduce features for LinearSVC.** The high-dimensional sparse space is precisely where LinearSVC excels.

### D. Cross-Validation Insights

1. **M2:** LogReg won 5-fold CV despite PA having higher single-split F1 (PA was overfitting).

2. **V7:** GridSearchCV for LinearSVC C values:
   - Best C = 0.08 (CV F1 = 0.8157)
   - C=0.01: 0.7903, C=0.1: 0.8150, C=0.5: 0.7984
   - Very sensitive around C=0.05-0.15

### E. Submission Pipeline (Best Practice from V5/V7)

1. Retrain on ALL training data (not just 80%)
2. Use same TF-IDF/scaler transformers
3. Generate multiple submissions (LinearSVC, Stacking, Seed-averaged)
4. **Seed averaging** (V7): train 5 models with different random seeds, average predictions → more robust

---

## 5. What Worked vs What Didn't

### ✅ What Worked
- LinearSVC with C=0.08-0.1 and custom/balanced weights
- Large TF-IDF (35k word + 25k char = 60k features)
- Char n-grams (char_wb, 3-5 grams)
- Stacking ensemble (4→LR meta-learner)
- class_weight handling (custom or balanced)
- Seed averaging for final predictions

### ❌ What Didn't Work
- KNN (0.56 F1 — terrible)
- Naive Bayes (0.63-0.67 — weak)
- SVD/dimensionality reduction for LinearSVC (hurts performance)
- Chi² feature selection with reduced K (drops from 0.82 to 0.71)
- SMOTE/ROS (inferior to class_weight)
- PA with large feature sets (0.76 in V7)
- GridSearchCV for LogReg (too slow, 800s/fit)

---

## 6. Best Achieved Scores

| Milestone | Best Model | Val F1 | CV F1 | Config |
|---|---|---|---|---|
| M1 | LogReg baseline | ~0.780 | — | 15k word |
| M2 | PA(C=0.01) | 0.809 | 0.792 | 35k (Config D) |
| M3 (submitted) | LinearSVC(C=0.1) | 0.870* | 0.813 | 25k word + 25k char |
| **M3 (our run)** | **LinearSVC(C=0.08)** | **0.826** | **0.822** | **60k features** |
| M5-V5 | Stacking | 0.827 | — | 60k features |
| M5-V7 | Stacking+SeedAvg | 0.827 | — | 60k, C=0.08 |
| **M4 (80k)** | **LinearSVC(custom)** | **0.828** | — | **80k features** |
| **M4 (60k)** | **LinearSVC(custom)** | **0.826** | **0.823** | **60k features** |
| **M5 Final** | **SeedAvg LinearSVC** | **~0.826** | **0.823** | **60k features** |

*M3 submitted 0.870 is a single-split val score (optimistic). CV was 0.813.

---

## 7. M4 Results (from partial + last runs)

### Weight Search (on 80k, ~5500s):
- balanced: F1=0.826 | V5_custom: 0.828 | aggressive: 0.828
- **moderate: 0.828 ← WINNER** | push_L1_L3: 0.827 | push_all: 0.826
- Winner: `{0: 1.0, 1: 4.5, 2: 2.0, 3: 12.0}`

### C-Value Search (moderate weights, 80k):
- C=0.05: 0.826 | C=0.08: 0.827 | **C=0.10: 0.828** | C=0.12: 0.828

### M4 Model Results (60k features):
| Model | Val F1 | Time |
|---|---|---|
| LinearSVC(custom) | **0.826** | 589s |
| Stacking(4→LR) | 0.825 | 8119s |
| Voting(soft) | 0.824 | 1128s |
| CalibSVC | 0.813 | 1173s |
| LogReg | 0.809 | 1343s |
| PA | 0.802 | 25s |
| SGD | 0.794 | 17s |

### M4 Failures:
- Pseudo-labeling: **OOM** (stacking retrain on 198k+60k samples exceeded 16GB)
- Stacking with n_jobs=-1: **OOM** (parallel workers each copy data)

---

## 8. M5 Final Strategy

M5 consolidates everything that works:
1. V5-style minimal text cleaning
2. 60k TF-IDF (35k word + 25k char) — proven sweet spot
3. Custom weights "moderate" `{0:1, 1:4.5, 2:2, 3:12}` — hardcoded from M4
4. LinearSVC(C=0.1) — hardcoded from M4
5. Seed-averaged predictions (5 seeds) — robust final submission
6. Comprehensive EDA + "What Didn't Work" section
7. Estimated runtime: ~3-4 hours
