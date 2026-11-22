# Milestone 5 — Final Explanation & Viva Preparation

> **Erratum:** this document describes the *course* M5 notebook (seed-averaged LinearSVC, ~0.826). The model submitted to Kaggle in the V1–V7 push was the **V5 stacking ensemble** (0.826); seed averaging was tested in V7 and did not help (-0.0006). Also, the 54 structured features mentioned here were reduced to 19 in V5. See `docs/project_retrospective.md`.

> **This is the final, submission-ready notebook.** It consolidates 9 milestones
> of experimentation into a clean, comprehensive pipeline.

---

## Full Milestone Progression

| Milestone | Best Model | Val F1 | CV F1 | Features | Key Innovation |
|---|---|---|---|---|---|
| M1 | LogReg(C=1.0) | 0.780 | — | 15k | Baseline TF-IDF |
| M2 | PA(C=0.01) | 0.809 | 0.792 | 35k | class_weight + char n-grams |
| M3 | LinearSVC(C=0.1) | 0.826 | 0.822 | 60k | SVM + bigger TF-IDF |
| M4 | LinearSVC(custom) | 0.826 | 0.823 | 60k | Custom weights + stacking |
| **M5** | **SeedAvg LinearSVC** | **~0.826** | **0.823** | **60k** | **Final consolidated** |

**Cumulative: +0.050 F1 (+6.4% relative) from M1 to M5**

---

## What the M5 Notebook Contains

### §1. EDA (Detailed)
- Class distribution (20.9:1 imbalance)
- Text length analysis by class
- Engagement features (upvote/downvote) by class
- Identity/demographic feature presence
- Emoticon feature analysis
- Full feature correlation matrix

### §2. Preprocessing
- V5-style minimal text cleaning (proven better than aggressive cleaning)
- 54 engineered structured features

### §3. TF-IDF (60k)
- 35k word (1,3)-grams + 25k char_wb (3,5)-grams
- No stop_words removal (V5 proved it helps)

### §4. Model Comparison (4 models)
- LinearSVC(custom weights, C=0.1) — CHAMPION
- LogReg(balanced, C=2.0)
- SGD(log_loss, balanced)
- PA(balanced, C=0.01)
- With classification reports and confusion matrix

### §5. Ensemble Methods
- Stacking (4 base → LR meta, n_jobs=1)
- Voting (soft, 3 models)

### §6. Cross-Validation
- 5-fold CV: LinearSVC(custom) and LinearSVC(balanced)
- Stacking CV skipped (would take ~7 hours)

### §7. Final Submission
- Seed-averaged LinearSVC (5 seeds)
- Single-model backup submission

### §8. What Didn't Work (with charts)
- KNN, NB, SVD, SMOTE, Chi², pseudo-labeling, stop_words
- Feature engineering experiments chart

### §9. Conclusions
- Full milestone progression
- Key takeaways

---

## Key Design Decisions

### Why LinearSVC over LogReg?
LinearSVC uses `liblinear` coordinate descent — O(n × features) per iteration.
LogReg with `lbfgs` uses quasi-Newton methods — O(n × features²) memory.
On 60k sparse features, LinearSVC is 3-5× faster AND gets better F1.

### Why custom weights {0:1, 1:4.5, 2:2, 3:12}?
sklearn's `balanced` gives: {0:0.43, 1:3.11, 2:0.79, 3:9.05}
Our custom "moderate" weights push harder on minority classes (Label 1 and 3)
and slightly on Label 2. This gained +0.002 F1 over balanced.

### Why seed averaging instead of stacking?
M4 showed LinearSVC(custom) = 0.826 > Stacking = 0.825.
Stacking doesn't help because:
- All base models are linear → low diversity
- The meta-learner can't learn much from 4 similar models
- Seed averaging gives equal robustness at 1/5 the cost

### Why 60k features, not more?
80k features gave marginal gain (+0.002) but caused OOM on Kaggle.
60k is the sweet spot: ~0.826 F1, comfortable memory, ~5 min vectorization.

---

## Runtime Estimate

| Section | Time |
|---|---|
| §1 EDA | ~5 min |
| §2-3 Preprocessing + TF-IDF | ~8 min |
| §4 Model comparison (4 models) | ~35 min |
| §5 Ensembles (stacking + voting) | ~60 min |
| §6 5-fold CV (2 models) | ~35 min |
| §7 Seed averaging + submissions | ~45 min |
| §8-9 Analysis/charts | ~2 min |
| **Total** | **~3-4 hours** |

---

## Viva Q&A

### Q: Why is Macro F1 the right metric, not accuracy?
**A:** With 20.9:1 imbalance, a model predicting all Label 0 gets 57.7% accuracy.
Macro F1 weights all classes equally — forcing the model to perform well even on
the 2.76% minority class (Label 3).

### Q: What's the mathematical difference between LinearSVC and LogReg?
**A:** Both find a linear hyperplane w·x + b = 0.
- LinearSVC minimizes **hinge loss**: max(0, 1 - y·f(x)) — margin-based, no probabilities
- LogReg minimizes **log loss**: log(1 + exp(-y·f(x))) — probabilistic, smoother
LinearSVC's hinge loss is more "decisive" — it gives up on outliers beyond the margin.

### Q: Why did SVD hurt LinearSVC?
**A:** SVD projects 60k sparse features → k dense dimensions by maximizing variance.
But **variance ≠ discrimination**. The TF-IDF features that distinguish Label 1 from
Label 0 might have low variance (rare words) but high discriminative power.
SVD kills these rare features → F1 drops significantly.

### Q: How does seed averaging work?
**A:** LinearSVC uses coordinate descent with random initialization. Different seeds →
different optimization paths → slightly different decision boundaries. By averaging
the decision_function (continuous values), we get a smoother, more robust boundary.
It's like bagging but without resampling — just different random starts.

### Q: Why didn't stacking help?
**A:** Stacking needs **diverse** base models. Our 4 models (SVM, LR, SGD, PA) are all
linear — they make similar mistakes. The meta-learner can't learn much from highly
correlated predictions. Stacking shines when mixing very different model families
(e.g., SVM + RandomForest + Neural Net).

### Q: What would you need to hit 0.87+ F1?
**A:** Based on leaderboard analysis, 0.87+ likely requires:
1. **Transformer models** (BERT/DistilBERT) that understand context
2. **Word embeddings** (Word2Vec/FastText) instead of TF-IDF
3. TF-IDF treats "not good" ≈ "good" — it lacks semantic understanding
4. Our ~0.826 is the **ceiling for TF-IDF + sklearn linear models**
