# 📊 Results

This folder contains experiment summaries and Kaggle submission files.

---

## `all_experiments_summary.md`

**The definitive record of every single experiment** across all 5 milestones.

Contains:
- Dataset statistics (197,999 train, 102,000 test, label distribution, imbalance ratio)
- All TF-IDF configurations tested (Config A through D, V5/V7 config)
- Complete model results table for every milestone
- Key insights: feature engineering, model comparison, dimensionality reduction, cross-validation
- What worked vs what didn't (with specific F1 numbers)
- Best achieved scores per milestone
- M4 weight search results and C-value tuning results
- M5 final strategy rationale

---

## `submissions/`

Kaggle submission CSV files — proof of actual competition runs.

| File | Model | Kaggle Macro F1 |
|---|---|:---:|
| `submission_final_stacking.csv` | Stacking (4→LR meta) | **0.826** |
| `submission_linearsvc.csv` | LinearSVC (C=0.1, balanced) | 0.822 |

### Submission Format
```
id,label
0,0
1,2
2,0
...
```

102,000 rows — one prediction per test comment.

---

## Score History

```
M1 Baseline (LogReg, 15k TF-IDF):      ~0.777
M2 Best (PA C=0.01, 35k config):        0.809 val / ~0.776 Kaggle
M3 LinearSVC (C=0.1, 60k):             0.826 val / 0.816 Kaggle
M4 Stacking (4→LR, 60k):               0.826 val / 0.823 Kaggle
M5 V5 Final (Stacking, 60k, clean):    0.826 val / 0.826 Kaggle ← FINAL
```

**Competition rank: ~139–157 out of 600+ participants**
