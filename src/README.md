# 🐍 Source Scripts (`src/`)

This folder contains **clean Python scripts** — the same logic as the notebooks, extracted into runnable `.py` files. These are production-quality implementations that can be run directly outside of Kaggle.

---

## Files

### `milestone_1_eda_preprocessing.py`
**Lines: ~979 | Sections: 22**

Complete EDA and baseline pipeline:
- Data loading with anti-leakage train/test separation
- Missing value analysis with custom `missing_report()` function
- Univariate analysis: triple-plot (histogram + box + Q-Q) for each numeric feature
- Bivariate analysis with Spearman correlation, Kruskal-Wallis tests, Cramér's V
- Text cleaning: `clean_text()` — lowercase, remove URLs, replace digits with `NUM`
- Feature engineering: vote ratios, identity flag aggregates, uppercase ratio, engagement features
- TF-IDF (15k, bigrams, `sublinear_tf=True`) + StandardScaler → hstack
- LogisticRegression baseline + 5-fold stratified CV
- Submission generation

---

### `milestone_2_linear_models_sgd.py`
**Lines: ~853 | Sections: 15**

Systematic linear model benchmark:
- Upgraded identity encoding (one-hot with `has_identity_data` binary)
- Log1p transforms for skewed features (if_1, if_2, upvote, downvote)
- RobustScaler (median/IQR) — robust to extreme outliers
- 4 TF-IDF configurations (A, B, C, D) evaluated systematically
- Models: LogReg, SGD (3 losses), Ridge, PassiveAggressive
- GridSearchCV and RandomizedSearchCV implementations
- SMOTE vs class_weight comparison with full analysis

---

### `milestone_3_nb_knn_svm.py`
**Lines: ~1064 | Sections: 18+**

Non-linear methods and feature engineering:
- TruncatedSVD with variance curve (shows 500 components = 27.2% variance)
- Chi-squared feature selection across K values: 10k, 20k, 30k, 40k, 50k, all
- MultinomialNB and ComplementNB with alpha tuning and MinMaxScaler for non-negativity
- KNN with 3 distance metrics (cosine, euclidean, manhattan) and k ∈ {3, 5, 7}
- LinearSVC C-value sweep: [0.01, 0.05, 0.08, 0.1, 0.15, 0.5]
- CalibratedClassifierCV (Platt scaling) for probability output
- 5-fold stratified CV confirming LinearSVC as champion

---

### `milestone_4_ensemble_final.py`
**Lines: ~1115 | Sections: 20+**

Ensemble methods with full experiment log:
- TruncatedSVD(300) → dense feature matrix for tree models
- GPU auto-detection: `nvidia-smi` → switch XGBoost to CUDA, LightGBM to GPU
- RandomForest, AdaBoost, XGBoost, LightGBM, MLP on dense features
- Custom class weight grid search (6 configurations)
- Stacking with `StackingClassifier` (4 base → LR meta, `cv=StratifiedKFold(3)`)
- Voting ensemble with soft probability averaging
- Seed averaging (5 seeds) using `decision_function` — not `predict`

---

### `milestone_5_final.py`
**Lines: ~600 | The definitive submission script**

Consolidated best-of-everything pipeline:
- V5-style minimal text cleaning (proven optimal)
- 60k TF-IDF (35k word (1,3)-grams + 25k char_wb (3,5)-grams)
- 54 structured features with RobustScaler
- 4-base stacking ensemble → LR meta-learner
- 5-fold stratified cross-validation
- Final retraining on ALL 197,999 training samples
- Multiple submission CSV generation

---

### `milestone_5_v5_clean.py`
**The cleanest, leanest version — 16s text cleaning, no illegal libraries**

Identical score to NLTK version (0.82631 Macro F1 to 5 decimal places), proving that complexity adds nothing. This is the reference implementation.

---

## Running Any Script

```bash
# Set up the data path at the top of any script:
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
SAMPLE_PATH = "data/Sample.csv"

# Run:
python src/milestone_5_final.py
```

**Runtime estimate:** ~3–4 hours on a machine with 16GB RAM.
