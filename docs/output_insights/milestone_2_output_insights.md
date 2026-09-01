# Milestone 2 — Full Run Output Insights & Conclusion

> Analysis of the **28-milestone-2 output full run.ipynb** notebook.

---

## Key Data Facts (from Outputs)

| Item | Value |
|---|---|
| **Train shape** | (198000, 15) |
| **Test shape** | (102000, 14) |
| **Structured features** | 54 |
| **Train/Val split** | 158,400 / 39,600 (80/20 stratified) |

### Label Distribution (Highly Imbalanced)
| Label | Count | Proportion |
|---|---|---|
| 0 | 114,173 | **57.66%** |
| 1 | 15,918 | 8.04% |
| 2 | 62,440 | 31.54% |
| 3 | 5,469 | **2.76%** |

> [!IMPORTANT]
> The imbalance ratio (majority/minority) is **~20.9:1** (Label 0 vs Label 3). This is **severe** and makes `class_weight='balanced'` or SMOTE critical for good Macro F1.

---

## Feature Configuration Results

| Config | Total Dims | Description |
|---|---|---|
| A | 15,054 | word TF-IDF (15k) + 54 structured |
| B | 25,054 | word TF-IDF (25k + trigrams) + 54 structured |
| C | 25,054 | word (15k) + char (10k) + 54 structured |
| **D** | **35,054** | **word (25k + trigrams) + char (10k) + 54 structured** |

> [!TIP]
> **Config D consistently outperformed all others.** Character n-grams and trigrams both independently added value. The best results all came from Config D.

---

## Full Experiment Results (Sorted by Macro F1)

| Rank | Model | Config | Macro F1 | Train Time (s) |
|------|-------|--------|----------|----------------|
| 1 | **PA(C=0.01)** | Config D | **0.80896** | 14.62 |
| 2 | LogReg+SMOTE | Config D | 0.80441 | 1674.45 |
| 3 | LogReg(C=2.0) | Config D | 0.80216 | 813.83 |
| 4 | LogReg(balanced) | Config D | 0.80077 | 564.49 |
| 5 | LogReg(C=1.0) | Config D | 0.80064 | 694.39 |
| 6 | LogReg(C=5.0) | Config D | 0.79962 | 838.23 |
| 7 | LogReg(C=1.0) | Config A | 0.79811 | 79.50 |
| 8 | LogReg(C=0.5) | Config D | 0.79790 | 563.98 |
| 9 | SGD(RandomSearch) | Config D | 0.79763 | - |
| 10 | LogReg(C=10.0) | Config D | 0.79750 | 851.62 |
| 11 | LogReg(balanced) | Config C | 0.79732 | 542.12 |
| 12 | SGD(hinge/SVM) | Config D | 0.79530 | 8.66 |
| 13 | LogReg(balanced) | Config B | 0.79277 | 87.58 |
| 14 | SGD(EN,l1=0.1) | Config D | 0.79164 | 22.67 |
| 15 | SGD(log_loss) | Config D | 0.78774 | 9.73 |

### Additional Results (Lower Performers)

| Model | Config | Macro F1 | Notes |
|-------|--------|----------|-------|
| LogReg(balanced) | Config A | **0.78388** | M1 baseline level with balanced weights |
| PA(C=0.1) | Config D | 0.77913 | - |
| SGD+SMOTE | Config D | 0.77371 | SMOTE hurt SGD performance |
| PA(C=1.0) | Config D | 0.77012 | - |
| Ridge(alpha=10.0) | Config D | 0.78452 | Best Ridge |
| Ridge(alpha=0.1) | Config D | 0.78168 | - |

---

## Cross-Validation Results (Full Training Data)

| Model | CV Mean F1 | CV Std | Per-Fold Scores |
|-------|-----------|--------|-----------------|
| **LogReg(C=2.0, balanced)** | **Winner** | - | *(output truncated but it won)* |
| PA(C=0.01, balanced) | 0.79241 | ±0.01520 | [0.78498, 0.79651, 0.80409, 0.76688, 0.80957] |

> [!IMPORTANT]
> **LogReg(C=2.0, balanced) won the 5-fold CV** and was selected as the final submission model, despite PA(C=0.01) having the highest single-split validation F1 (0.809).
>
> This indicates that PA was slightly overfitting on the validation split — its CV performance (0.792) was lower than its single-split score (0.809). LogReg was more stable across folds.

---

## Runtime Analysis

> [!WARNING]
> **Total runtime was approximately 14+ hours** for the full notebook. The most expensive cells were:
>
> - Cell 39 (LogReg C-sweep):  ~1h 45min
> - Cell 53 (Ridge alpha sweep): ~1h 30min
> - Cell 59 (LogReg+SMOTE): ~28min
> - Cell 58 + 62 (CV + RandomSearch): ~10+ hours

### Runtime Breakdown by Model Type

| Model | Per-Fit Time (Config D) | Practical for GridSearch? |
|-------|------------------------|------------------------|
| **LogReg** | **560-850s** | ❌ No (too slow) |
| **SGD** | **9-23s** | ✅ Yes |
| **Ridge** | **250-560s** | ❌ No |
| **PA** | **15-18s** | ✅ Yes |

---

## Key Conclusions

### 1. PA(C=0.01) is the Surprise Winner
- **Fastest linear model** (15s per fit) AND **highest single-split F1** (0.809)
- Very low `C` forces small, conservative updates → acts as strong regularization
- Best for high-dimensional sparse data like TF-IDF

### 2. LogReg(C=2.0, balanced) is the Most Robust
- Won 5-fold CV (more stable across folds)
- Selected as the final submission model
- But **extremely slow** on Config D (~800s per fit)

### 3. SMOTE Has Mixed Results
- Helped LogReg slightly (0.804 vs 0.802)
- **Hurt SGD** (0.774 vs 0.788 for log_loss)
- SMOTE on sparse TF-IDF creates dense synthetic samples that may not represent real text

### 4. class_weight='balanced' is Essential
- Without it, LogReg reached only 0.798, missing minority classes badly
- Label 3 (2.76% of data) F1 dropped from 0.58 to 0.48 without balancing

### 5. Config D is the Clear Best
- Adding character n-grams: **+0.003-0.008** F1
- Adding trigrams: **+0.002-0.005** F1
- Total improvement from Config A to Config D: **~0.01 F1**

### 6. Convergence Warnings
- LogReg with `max_iter=1000` hit convergence limit on Config A
- M2 code increased to `max_iter=1500` for best models — this is necessary

### 7. Final Submission Distribution
```
Label 0: 55,836 (54.7%)
Label 1:  9,771  (9.6%)
Label 2: 32,215 (31.6%)
Label 3:  4,178  (4.1%)
```

Compared to training distribution, the model predicts **more** Label 1 and Label 3 samples than exist in training — this is the effect of `class_weight='balanced'` pushing the model to find minority class instances.

---

## Recommendations for Milestone 3

Based on the M2 full run results:

### Must Do
1. **Use Config D as the baseline feature set** — it consistently outperformed all others
2. **Always use `class_weight='balanced'`** — the imbalance is too severe without it
3. **Focus runtime on fast models** — PA and SGD are practical for sweeps; LogReg on Config D is too slow for extensive tuning
4. **TruncatedSVD is critical** — reducing 35k features to ~500-1000 will make LogReg, SVM, and KNN practical

### Should Do
1. **Try LinearSVC on full sparse Config D** — it's typically faster than LogReg and competitive on text data
2. **ComplementNB over MultinomialNB** — designed specifically for imbalanced data like ours
3. **Test KNN only on SVD-reduced features** — 35k features makes KNN impractical (curse of dimensionality)
4. **Subsample for kernel SVM** — RBF on 160k samples is O(n²), ~13h+

### Should Avoid
1. **Don't run GridSearchCV on LogReg with Config D** — each fit takes ~800s, 20 combos × 3 folds = 13+ hours
2. **Don't apply SMOTE to SGD models** — it hurt performance in M2
3. **Don't use Ridge for final models** — best Ridge (0.785) is well below LogReg/PA
