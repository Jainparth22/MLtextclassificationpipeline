# Milestone 4 — Explanation & Viva Preparation

> This document explains every design decision in the Milestone 4 notebook.

---

## Summary of ALL Previous Results

| Milestone | Best Model | Val F1 | CV F1 | Features | Key Technique |
|---|---|---|---|---|---|
| M1 | LogReg(C=1.0) | 0.780 | — | 15k | Baseline TF-IDF |
| M2 | PA(C=0.01) | 0.809 | 0.792 | 35k | class_weight, char n-grams |
| M3 | LinearSVC(C=0.1) | 0.826 | 0.822 | 60k | SVM + larger TF-IDF |
| V5 sub | Stacking(4→LR) | 0.827 | — | 60k | Custom weights + ensemble |
| V7 sub | Stacking+SeedAvg | 0.827 | — | 60k | C=0.08 + seed averaging |
| **M4** | **All combined** | **Target: 0.87+** | — | **80k** | **Everything below** |

---

## What M4 Does Differently (and Why)

### 1. V5-Style Minimal Text Cleaning

**M2/M3 cleaning:**
```python
text = re.sub(r'http\S+|www\.\S+', '', text)  # remove URLs
text = re.sub(r'@\w+', '', text)               # remove @mentions
text = re.sub(r'\d+', ' NUM ', text)           # replace numbers
```

**M4 cleaning (V5-style):**
```python
text = text.lower()
text = re.sub(r'[^a-z0-9\s\'\"!?.,;:()\-]', ' ', text)  # keep alphanum + punctuation
text = re.sub(r'\s+', ' ', text).strip()
```

**Why this is better:**
- **URLs contain signal:** A comment with many URLs might be spam (Label 1)
- **Numbers contain signal:** "$100,000" vs "3 years" indicates topic
- **@mentions contain signal:** Tagging behavior correlates with comment type
- V5 used minimal cleaning and achieved 0.822+ F1

### 2. Larger TF-IDF (50k word + 30k char = 80k)

| Config | Word | Char | Total | Best F1 |
|---|---|---|---|---|
| M2 Config D | 25k (1,3) | 10k (3,5) | 35k | 0.809 |
| M3 | 35k (1,3) | 25k (3,5) | 60k | 0.826 |
| **M4** | **50k (1,3)** | **30k (2,6)** | **80k** | **TBD** |

**Key changes:**
- **Word TF-IDF: 50k** — captures more rare but discriminative terms
- **Char n-grams: (2,6)** — wider range captures more patterns. (2,6) includes bigrams ("th", "he") AND 6-grams that capture word fragments
- **No `stop_words='english'`** — V5 proved this helps. TF-IDF's `sublinear_tf` and `max_df=0.95` already downweight common words

### 3. Custom Class Weights

sklearn's `class_weight='balanced'` computes:
```
weight_class_i = n_total / (n_classes × n_class_i)
```
→ {0: 0.43, 1: 3.11, 2: 0.79, 3: 9.05}

**V5's custom weights pushed harder:**
```python
CUSTOM_WEIGHTS = {0: 1.0, 1: 5.5, 2: 1.8, 3: 14.0}
```

**Why custom > balanced:**
- Macro F1 weights ALL classes equally, but some classes are harder
- Label 3 (2.76%) needs extra weight because it has very few samples
- Label 1 (8%) is also minority but has more distinctive features
- The optimal weights depend on the specific confusion patterns

**We grid-search 6 weight configurations** to find the best.

### 4. Stacking Ensemble

**Architecture:**
```
Base Models (diverse):
  ├─ CalibratedSVC (balanced) → probabilities
  ├─ LogReg (balanced) → probabilities
  ├─ SGD (balanced) → probabilities
  └─ PA (balanced) → probabilities
        ↓
  [4×4 = 16 probability features]
        ↓
Meta-Learner: LogisticRegression
        ↓
  Final Prediction
```

**Why stacking works:**
- **Diversity:** Each model learns different decision boundaries
- **Error correction:** Where LinearSVC is wrong, LogReg might be right
- **Optimal combination:** The meta-learner finds the best weights automatically
- V5/V7 confirmed Stacking = +0.005 over best single model

**Why these 4 base models:**
1. **CalibratedSVC:** Best individual model, linear SVM approach
2. **LogReg:** L-BFGS optimization, probabilistic, different from SVM
3. **SGD:** Online learning, different convergence path
4. **PA:** Aggressive updates, different error sensitivity

### 5. Pseudo-Labeling (Semi-Supervised)

**How it works:**
1. Train stacking on ALL labeled data
2. Predict test set → get probability estimates
3. Keep only predictions with ≥95% confidence
4. Add these as "pseudo-labeled" training samples
5. Retrain on augmented dataset

**Why it helps:**
- Adds ~60k training samples (from V5: 63.4% of test at 95% threshold)
- The test distribution may differ from training → model learns test patterns
- Only high-confidence samples minimize label noise

**Risk:** If the model is systematically wrong on some class, pseudo-labels amplify the error. The 95% threshold mitigates this.

### 6. Seed Averaging

**How it works:**
```python
for seed in [42, 123, 456, 789, 2024]:
    model = LinearSVC(random_state=seed)
    model.fit(X_train, y_train)
    predictions.append(model.decision_function(X_test))
final = np.mean(predictions, axis=0).argmax(axis=1)
```

**Why it helps:**
- LinearSVC with `liblinear` uses coordinate descent → different random seed → different optimization path → slightly different model
- Averaging the decision functions smooths out random fluctuations
- More robust than any single model

---

## Runtime Estimate

| Section | Operation | Est. Time |
|---|---|---|
| §1-3 | Preprocessing + features | ~30s |
| §4 | TF-IDF (80k features) | ~5 min |
| §6 | Custom weight search (6 configs) | ~1 hour |
| §7 | C-value tuning (6 values) | ~1 hour |
| §8 | 5 base models | ~40 min |
| §9 | Stacking ensemble | ~30 min |
| §10 | Voting ensemble | ~20 min |
| §11 | 5-fold CV (3 models) | ~3 hours |
| §12 | Pseudo-labeling | ~30 min |
| §14 | Final retraining + seed avg | ~30 min |
| **Total** | | **~7-8 hours** |

Within Kaggle's 12-hour limit.

---

## Viva Q&A

### Q: Why is stacking better than voting?
**A:** Voting gives equal weight to all models (or manually set weights). Stacking LEARNS the optimal combination — it trains a meta-learner on out-of-fold predictions. If Model A is great on Label 0 but bad on Label 3, and Model B is the opposite, stacking can learn to trust A for Label 0 and B for Label 3.

### Q: Why not use more base models in the stack?
**A:** Diminishing returns + runtime. Each additional model adds training time but less marginal diversity. The 4 models we chose represent fundamentally different optimization algorithms (SVM, quasi-Newton, online SGD, aggressive online). Adding another linear model wouldn't add much diversity.

### Q: How do custom class weights mathematically affect the loss?
**A:** In LinearSVC, the loss function becomes:
```
L = C × Σ w_class(i) × max(0, 1 - y_i × f(x_i)) + ½ ‖w‖²
```
Higher w_class means misclassifying that class is more expensive → the model dedicates more capacity to getting minority classes right, at the cost of majority class accuracy.

### Q: Why is pseudo-labeling risky?
**A:** It's a form of **confirmation bias**. The model predicts test labels → you train on those predictions → the model becomes more confident in its original predictions. If the model was systematically wrong (e.g., confusing Label 1 with Label 0), pseudo-labeling reinforces the error. The 95% threshold mitigates this by only using the most confident predictions.

### Q: Why use decision_function for seed averaging instead of predict?
**A:** `predict()` returns hard labels (0/1/2/3) — averaging labels is meaningless (what's the average of 0 and 3?). `decision_function()` returns continuous scores for each class — averaging these is meaningful and preserves the model's confidence information.

### Q: How does `passthrough=False` in stacking work?
**A:** With `passthrough=False`, the meta-learner only sees the base models' probability predictions (4 models × 4 classes = 16 features). With `passthrough=True`, it also sees the original features (80k+). We use False because: (1) the meta-learner would overfit to 80k features, (2) runtime increases dramatically, and (3) the base models already capture the feature information.

### Q: What's the total improvement from M1 to M4?
**A:** M1 (0.780) → M4 (target 0.87+) = **+0.09 Macro F1** (+11.5% relative). The improvements came from: better features (+0.03), better model family (+0.02), custom weights (+0.01), ensembling (+0.005), pseudo-labeling (+0.01), and cumulative optimizations.
