<div align="center">

# 🧠 Toxic Comment Category Prediction
### Multi-Class NLP Classification · Kaggle Competition · Top ~150 of 600+ teams

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Macro%20F1-0.826-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/)
[![Python](https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Summary

This repository documents my complete, end-to-end ML journey for a Kaggle competition on **multi-class toxic comment classification**. What started as a course assignment became a deep research process — 5 milestones, 10+ notebook iterations, 12 models tested, and a final score of **0.826 Macro F1** placing me in the **top ~25%** of competitors.

> **No shortcuts. No AutoML. No transformers.** Only raw scikit-learn, mathematical intuition, and systematic experimentation — the way ML is actually learned.

---

## 🎯 The Problem

Classify online comments into **4 toxicity categories**:

| Label | Meaning | Train % |
|:---:|---|:---:|
| 0 | Neutral / benign | 57.7% |
| 1 | Mildly problematic | 8.0% |
| 2 | Moderately toxic | 31.5% |
| 3 | **Severely toxic** | **2.7%** ← hardest class |

**Key challenges:**
- 🔴 **Severe class imbalance** — 20.9:1 ratio (Label 0 vs Label 3)
- 📊 **Metric:** Macro F1 — treats all 4 classes equally, no hiding behind majority-class accuracy
- ⚖️ **Library constraint** — No BERT, PyTorch, TensorFlow, NLTK, spaCy. Only `sklearn`, `scipy`, `xgboost`, `lightgbm`, `imblearn`

**Dataset:** 197,999 train rows × 15 features · 102,000 test rows · Text + engagement + demographic features

---

## 📈 Score Progression (Milestones 1 → 5)

```
Milestone 1  ████░░░░░░░░░░░░  0.777  LogReg baseline (15k TF-IDF)
Milestone 2  ███████░░░░░░░░░  0.776* SGD, Ridge, RobustScaler, char n-grams
Milestone 3  ████████████░░░░  0.816  LinearSVC — BIGGEST jump (+0.04)
Milestone 4  █████████████░░░  0.823  Stacking Ensemble
Milestone 5  ██████████████░░  0.826  Final push (7 versions) ← FINAL SUBMISSION
```
*M2 Kaggle score slightly lower due to config; single-split val F1 was 0.809

**Total improvement: +0.049 Macro F1 (+6.3% relative)** across 5 milestones and hundreds of experiments.

---

## 🏗️ Final Architecture — What Won

### Feature Pipeline (60,054 total dimensions)

```python
# 1. Text Cleaning — minimal by design (less is more, proven by experimentation)
text = text.lower()
text = re.sub(r'[^a-z0-9\s\'\"!?.,;:()\-]', ' ', text)

# 2. TF-IDF (the dominant signal — 60k features)
word_tfidf = TfidfVectorizer(max_features=35_000, ngram_range=(1,3), sublinear_tf=True, analyzer='word')
char_tfidf = TfidfVectorizer(max_features=25_000, ngram_range=(3,5), sublinear_tf=True, analyzer='char_wb')

# 3. 54 Structured features: engagement (votes, ratios), demographics,
#    text stats (length, word count, uppercase ratio), datetime, post frequency

# 4. Scaling: RobustScaler (median/IQR — robust to if_1/if_2 extreme outliers)
```

### Stacking Ensemble — The Winning Model

```
Base Estimators (4 diverse linear models, all on full 60k sparse features):
  ┌─ CalibratedClassifierCV(LinearSVC(C=0.1, class_weight='balanced'))
  ├─ LogisticRegression(C=1.0, class_weight='balanced')
  ├─ SGDClassifier(loss='log_loss', alpha=1e-5, class_weight='balanced')
  └─ CalibratedClassifierCV(PassiveAggressiveClassifier(C=0.1))
         │
         ▼  [4 models × 4 classes = 16 probability features]
         │
  Meta-Learner: LogisticRegression(C=1.0)
         │
         ▼
  Final Prediction → Kaggle Macro F1: 0.826
```

---

## 📊 All Models Tested — Complete Results

| Model | Best Val F1 | Verdict |
|---|:---:|---|
| **Stacking (4 base → LR meta)** | **0.826** | 🏆 Champion |
| Voting Ensemble (4-model soft) | 0.825 | Near-best |
| LinearSVC (custom class weights) | 0.822 | Best individual model |
| SGDClassifier (log_loss) | 0.812 | Fast and competitive |
| LogisticRegression | 0.808 | Slowest (1300s/fit) |
| CalibratedClassifierCV(SVC) | 0.807 | Needed for ensembling |
| PassiveAggressiveClassifier | 0.764 | Fell off with 60k features |
| RidgeClassifier | 0.780 | Weak for this task |
| LightGBM (SVD-reduced) | 0.757 | SVD information loss kills it |
| MultinomialNB | ~0.650 | Independence assumption too naive |
| ComplementNB | 0.622 | Hurt ensemble diversity |
| **KNN (SVD 200-dim)** | **~0.600** | **Curse of dimensionality — worst** |

---

## 🔬 Key Technical Decisions & Why They Worked

### ✅ What Worked

| Decision | Why |
|---|---|
| **LinearSVC over LogReg** | `liblinear` coordinate descent O(n×d) vs LBFGS O(n×d²) — 3–5× faster AND better F1 |
| **60k TF-IDF (word + char)** | Char n-grams capture censored profanity ("f\*\*k"), misspellings, morphological patterns |
| **No stop word removal** | Words like "not", "you" carry sentiment context — removing them slightly hurt |
| **No stemming / lemmatization** | TF-IDF trigrams already capture morphology; over-stemming ("tribal"→"trib") hurt scores |
| **class_weight='balanced'** | Mathematically penalises minority class errors more — directly aligned with Macro F1 |
| **Custom weights {0:1, 1:4.5, 2:2, 3:12}** | Pushes harder on rarest classes than sklearn's auto-balanced formula |
| **Stacking over Voting** | Meta-learner learns optimal combination; voting applies fixed equal weights |
| **LR meta-learner** | Only 16 probability inputs — nonlinear meta (LightGBM) overfits. LR simplicity wins |

### ❌ What Didn't Work (and Why It's Important to Know)

| Technique | Why It Failed |
|---|---|
| SMOTE | Interpolating 60k-dim sparse TF-IDF vectors creates unrealistic synthetic text |
| RandomOverSampler + class_weight | Double correction — penalty applied twice → overcorrection |
| TruncatedSVD (dimensionality reduction) | 500 SVD components = only 27.2% variance. Kills LinearSVC — sparse space IS the signal |
| Chi² feature selection | Distorted by `abs()` on StandardScaled features → selected wrong features |
| LightGBM as meta-learner | Overfit on 16 probability features; scored 0.787 vs LR meta's 0.826 |
| Pseudo-labeling | OOM crash + amplified majority class bias in high-confidence predictions |
| Threshold tuning | Always worse — 0.811 vs 0.818 |
| Seed averaging (LinearSVC) | Linear models too stable; averaging near-identical predictions adds no diversity |

---

## 🗂️ Repository Structure

```
.
├── 📓 notebooks/               ← Jupyter notebooks for each milestone
│   ├── milestone_1_eda_baseline.ipynb
│   ├── milestone_2_linear_models.ipynb
│   ├── milestone_3_nb_knn_svm.ipynb
│   ├── milestone_4_ensemble.ipynb
│   └── milestone_5/            ← 6 versioned iterations of the final push
│       ├── v1_feature_expansion.ipynb
│       ├── v3_hybrid_strategy.ipynb
│       ├── v4_max_attempt.ipynb       ← "kitchen sink" — everything that failed
│       ├── v5_clean_final.ipynb       ← ⭐ SUBMITTED — best score
│       ├── v6_diversity_experiment.ipynb
│       └── v7_final_tuning.ipynb
│
├── 🐍 src/                     ← Clean Python scripts (runnable outside Kaggle)
│   ├── milestone_1_eda_preprocessing.py
│   ├── milestone_2_linear_models_sgd.py
│   ├── milestone_3_nb_knn_svm.py
│   ├── milestone_4_ensemble_final.py
│   └── milestone_5_final.py
│
├── 📚 docs/                    ← Deep technical documentation for every milestone
│   ├── milestone_1_explanation.md     ← Cell-by-cell decisions + viva Q&A
│   ├── milestone_2_explanation.md
│   ├── milestone_3_explanation.md
│   ├── milestone_4_explanation.md
│   ├── milestone_5_explanation.md
│   ├── project_retrospective.md       ← Complete research logbook (10 versions)
│   └── output_insights/               ← Actual model outputs with analysis
│       ├── milestone_1_output_insights.md
│       ├── milestone_2_output_insights.md
│       └── milestone_3_output_insights.md
│
├── 📊 results/                 ← Experiment summary + submission files
│   ├── all_experiments_summary.md     ← Every model, every config, every score
│   └── submissions/
│       ├── submission_final_stacking.csv   ← Best score (0.826)
│       └── submission_linearsvc.csv        ← Best individual model
│
├── 🎓 viva_prep/               ← Exam preparation (proves theoretical depth)
│   └── viva_guide.md                  ← Q&A on every algorithm and decision
│
├── 📁 data/                    ← Empty (data too large for GitHub — see below)
│   └── .gitkeep
│
└── .gitignore
```

---

## 🚀 Reproducing Results

### 1. Get the Data
Download from Kaggle: [Comment Category Prediction Challenge](https://www.kaggle.com/)  
Place `train.csv`, `test.csv`, `Sample.csv` in the `data/` folder.

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn scipy imbalanced-learn matplotlib seaborn wordcloud
```

### 3. Run the Final Pipeline
```bash
# Option A: Run as Python script
python src/milestone_5_final.py

# Option B: Open the final notebook
jupyter notebook notebooks/milestone_5/v5_clean_final.ipynb
```

Expected runtime: **~3–4 hours** (on Kaggle free tier T4 GPU / ~16GB RAM).  
Final output: `submission_final_stacking.csv` with Macro F1 ≈ **0.826**

---

## 💡 10 Lessons Learned (The Real ML Education)

1. **Simplicity wins** — The simplest version (V5) matched every complex alternative. Every addition of complexity in V4/V6 either hurt or had zero effect.
2. **Lemmatization = zero value** — Identical Kaggle scores to 5 decimal places (0.82631) with and without NLTK. TF-IDF trigrams already capture morphology.
3. **Ensemble diversity requires quality** — Adding a weak model (ComplementNB, F1=0.622) dragged the whole ensemble down. All base models must be individually strong.
4. **Non-linear meta-learners overfit** — LightGBM meta scored 0.787 vs LR meta's 0.826 on only 16 input features. LR's simplicity is a strength here.
5. **SMOTE fails in high-dimensional sparse space** — Synthetic interpolation between 60k-dim TF-IDF vectors creates documents that don't resemble real text.
6. **Double correction kills** — ROS + class_weight='balanced' applies the imbalance penalty twice → catastrophic overcorrection.
7. **Feature engineering can't beat TF-IDF** — Hand-crafted "toxicity features" (caps_ratio, toxic_word_count) added negligible signal because trigrams implicitly capture them.
8. **C=0.1 was near-optimal from M3** — GridSearchCV across 8 C values confirmed the original intuitive choice was within 0.0006 of the optimum.
9. **Seed averaging doesn't help stable models** — Linear models produce near-identical predictions across random seeds. There's nothing to average.
10. **The 0.826 ceiling is real** — TF-IDF cannot understand sarcasm, negation scope, or implicit toxicity. Breaking 0.84+ requires BERT — which was disallowed.

---

## 📉 The 0.826 Ceiling Explained

```
Our ceiling:  0.826  (TF-IDF + linear models — exhaustively optimised)
Top 10:       0.845  (likely using contextual embeddings — possibly against rules)
Gap:          0.019  Macro F1

To cross this gap requires:
  → BERT / DistilBERT fine-tuning   (+0.05–0.10 expected)
  → Sentence embeddings             (+0.03–0.08 expected)
  → FastText / Word2Vec             (+0.01–0.03 expected)

These were disallowed by competition rules. Within the allowed library set,
0.826 is the theoretical maximum achievable — this is an empirical finding,
not a limitation of effort.
```

---

## 📖 Documentation Guide for Reviewers

If you want to go deep, here's the reading order:

| What you want to understand | Where to look |
|---|---|
| Full project story & all decisions | [`docs/project_retrospective.md`](docs/project_retrospective.md) |
| M1 EDA decisions (cell by cell) | [`docs/milestone_1_explanation.md`](docs/milestone_1_explanation.md) |
| Why LinearSVC beat everything | [`docs/milestone_3_explanation.md`](docs/milestone_3_explanation.md) |
| Ensemble design & failures | [`docs/milestone_4_explanation.md`](docs/milestone_4_explanation.md) |
| Every model's actual output & analysis | [`results/all_experiments_summary.md`](results/all_experiments_summary.md) |
| Algorithm theory Q&A | [`viva_prep/viva_guide.md`](viva_prep/viva_guide.md) |

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---|---|---|
| `scikit-learn` | 1.3+ | All ML models, TF-IDF, preprocessing |
| `pandas` | 2.0+ | Data manipulation |
| `numpy` | 1.24+ | Numerical operations |
| `scipy` | 1.10+ | Statistical tests (Kruskal-Wallis, Cramér's V) |
| `imbalanced-learn` | 0.11+ | SMOTE (tested, not used in final) |
| `matplotlib` + `seaborn` | Latest | EDA visualizations |
| `wordcloud` | 1.9+ | Text frequency visualization |
| `xgboost` / `lightgbm` | Latest | Tested in M4/M5; did not beat linear models |

---

## 👤 About This Project

This was a **graded Kaggle competition** run as part of a Machine Learning Practicum course. Each milestone had a formal submission deadline and viva (oral examination) where every design decision had to be defended in front of faculty.

The depth of documentation in this repo reflects not just code, but the ability to:
- **Justify every hyperparameter choice** with mathematical reasoning
- **Run controlled experiments** (change one variable at a time)
- **Interpret failure** as rigorously as success
- **Know when to stop** adding complexity

> *"Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away."* — Antoine de Saint-Exupéry  
> V5 embodies this: the simplest pipeline that achieves the maximum possible score.

---

<div align="center">

**⭐ If this project helped you understand classical NLP or ML experimentation, consider starring the repo!**

</div>
