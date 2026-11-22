# Model Card — Toxic Comment Classifier (4-class)

## Summary
Stacking ensemble over sparse TF-IDF (word 1-3-gram + char_wb 3-5-gram) and 19 metadata features.
Predicts one of: 0 neutral, 1 mildly problematic, 2 moderately toxic, 3 severely toxic.

## Intended use
Triage / ranking of user comments for human review in a moderation workflow, as a research and portfolio artifact.
**Not intended** for automated punishment, account actions, or any decision without human review.

## Training data
Competition dataset: 197,999 labelled comments (57.7% / 8.0% / 31.5% / 2.7%). Includes metadata columns
(votes, emoticons, timestamps, identity-mention flags). Annotation process and annotator demographics are not documented.

## Performance
Fill from `results/metrics/SUMMARY.md` (leak-free): hold-out Macro-F1 with 95% CI, per-class precision/recall/F1,
5-fold CV mean +/- std. Kaggle public leaderboard: 0.826 (original notebooks, TF-IDF fit before split).
Class 3 (severe) is the weakest class; see `error_analysis_label3.csv`.

## Fairness considerations
* False-positive rate on neutral comments is reported per identity group in `fairness_audit.csv`.
* Identity flags are provided in the data and also used as a model input (`has_detection`), which can encode
  topic/identity correlations with toxicity labels. Subgroups are small; gaps are indicative, not conclusive.
* Models trained on toxicity labels commonly over-flag benign comments that mention identity terms or use dialects
  underrepresented in training data. Validate on your own population before relying on it.

## Limitations
Bag-of-words: no handling of sarcasm, negation scope, quoted/reported speech or context across a thread.
English only. Language and slang drift over time; retraining and monitoring are required.
Probabilities come from Platt-calibrated base models and a LogReg meta-learner but were not re-calibrated on a
separate set; do not treat them as exact risk estimates.

## Operational notes
Inference cost: 4 base models per request (latency numbers in `holdout_summary.md`). Artifact is a pickled sklearn
Pipeline, bound to the exact library versions in `model_metadata.json`; load only artifacts you built.

## Monitoring recommendations
Track input length/vocabulary drift, predicted class mix, and class-3 recall on a periodically human-labelled sample;
retrain when recall or the identity-group FPR gap degrades.
