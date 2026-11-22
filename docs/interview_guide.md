# Interview Guide: Toxic Comment Classification Project

Rule: never claim more than the evidence. Say whether a number is leaderboard, single split, or CV.

## 30-second pitch
"I built a four-class toxicity classifier on 198k comments with a 21:1 imbalance, optimising Macro-F1. Under a
no-neural-nets constraint I reached 0.826 on the Kaggle leaderboard with a stacked ensemble of linear models on 60k
sparse TF-IDF features. After the competition I audited my own work: I found a validation leak and fixed it with
in-fold feature fitting, added CV with confidence intervals, ablations, error analysis and a per-identity
false-positive audit, compared against DistilBERT to measure what the constraint cost, and shipped the model as a
versioned FastAPI container with tests and CI."

## Resume bullets (fill the brackets from `results/metrics/SUMMARY.md`)
* Built a 4-class toxic-comment classifier (198k comments, 21:1 imbalance); stacked 4 linear models on 60k sparse
  word+char TF-IDF features to reach **0.826 Macro-F1** on the Kaggle leaderboard (about top 25% of 600+ teams).
* Re-validated with leak-free stratified CV and paired-bootstrap CIs ([CV mean +/- std]); ablations showed
  [char n-grams: +x, structured features: +y]; a bias audit found [x pp] false-positive gap across [group].
* Quantified accuracy/latency/cost trade-offs: stack vs LinearSVC ([x ms vs y ms]) and DistilBERT ([macro-F1],
  [ms], [MB]).
* Shipped a pinned, containerised FastAPI service (single sklearn Pipeline artifact, provenance endpoint) with
  pytest and GitHub Actions CI.

## Questions to expect, with answer skeletons
**Why Macro-F1 and class weights?** Accuracy was 0.89 while class-3 recall was about 0.40. Macro-F1 gives each class
equal weight. Class weights re-weight the loss without inventing samples (SMOTE interpolates in 60k sparse
dimensions, producing unrealistic points).

**Why LinearSVC / linear models?** Text is high-dimensional, sparse and close to linearly separable. Hinge loss with
L2 is a strong baseline and coordinate descent scales. SVD plus trees lost most of the variance (observed 0.75-0.80
vs 0.82).

**Is stacking worth it?** It added about 0.003-0.004 on the leaderboard, within noise of single splits, at roughly 8x
the inference cost. I would ship LinearSVC unless the paired-bootstrap CI of the difference excludes 0 (I measured this).

**What was wrong with your validation?** TF-IDF and the scaler were fit before splitting. Unsupervised, so likely a
small effect, but I fixed it with in-fold fitting and re-ran: results [within noise / changed by x].

**What failed and why?** Separate bugs from findings. Chi² was a bug (`abs()` on scaled features). SMOTE and SVD
failures have mechanisms. The LightGBM meta-learner overfit 16 inputs. Do not present the Chi² result as a property of Chi².

**What does the model get wrong?** Show 3-4 examples from `error_analysis_label3.csv` by category (sarcasm,
negation, short or implicit). Explain why bag-of-words cannot fix them and what could (context models).

**Fairness?** Macro-F1 hides who is mis-flagged. I measured the false-positive rate on truly neutral comments per
identity group [numbers]. Caveats: provided flags, small groups, `has_detection` is an input. Mitigations: remove
the feature and re-measure, group-aware thresholds, more diverse data, human review.

**Transformer vs this?** [DistilBERT result]. Trade-off: +x F1 for y ms and z MB. For a moderation queue I would
use the cheap model to triage and the transformer on uncertain cases (a cascade).

**How would you deploy and monitor it?** Container, registry, then Container Apps or an Azure ML endpoint; metadata
endpoint for provenance; drift monitoring; sampled human labels for class-3 recall; promote a candidate only after
a paired comparison on a frozen set.

**Scale to 10M rows?** Stateless HashingVectorizer plus `partial_fit` SGD (memory fixed by `n_features`). Trade-offs:
no IDF, collisions, no balanced `class_weight` in `partial_fit` (use sample weights). See `docs/scalability_hashing.md`.

**What would you do differently?** Use a Pipeline from day one (no leak), CV plus CIs before deciding, track
experiments (MLflow), set a latency/cost budget up front, include fairness checks from the start.

## Traps to avoid
* Saying "0.826 is the maximum possible". Say "the best I reached with this feature family".
* Quoting 0.870 (M3 validation): it was an optimistic single split; CV was about 0.813.
* Claiming results you have not run (served-model score, DistilBERT, hashing) until the Colab outputs exist.
