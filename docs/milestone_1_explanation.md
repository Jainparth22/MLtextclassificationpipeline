# Milestone 1 — Complete Explanation & Decision Rationale

> **Purpose:** This document explains every cell, function, and design decision in `milestone_1_eda_preprocessing.py` so you can confidently defend each choice in front of an examiner. Read this alongside the notebook.

---

## Table of Contents
1. [Cell 0 – Imports](#cell-0--imports)
2. [Cell 1 – Data Loading](#cell-1--data-loading)
3. [Cell 2–5 – Initial Inspection](#cells-25--initial-inspection)
4. [Cell 6–7 – Missing Value Audit](#cells-67--missing-value-audit)
5. [Cell 8 – Duplicate Detection](#cell-8--duplicate-detection)
6. [Cell 9 – Target Variable Analysis](#cell-9--target-variable-analysis)
7. [Cells 10–12 – Univariate: Numeric Features](#cells-1012--univariate-numeric-features)
8. [Cells 13–15 – Univariate: Text Feature](#cells-1315--univariate-text-feature)
9. [Cells 16–18 – Datetime Exploration](#cells-1618--datetime-exploration)
10. [Cells 19–22 – Bivariate Analysis](#cells-1922--bivariate-analysis)
11. [Cell 23 – Cramér's V](#cell-23--cramérs-v)
12. [Cell 24 – Multivariate / Pair Plot](#cell-24--multivariate--pair-plot)
13. [Cells 25–26 – Outlier Detection](#cells-2526--outlier-detection)
14. [Cell 27 – Post-ID Analysis](#cell-27--post-id-analysis)
15. [Cells 28–33 – Feature Engineering](#cells-2833--feature-engineering)
16. [Cell 34 – Feature Matrix Preparation](#cell-34--feature-matrix-preparation)
17. [Cell 35 – Train-Validation Split](#cell-35--train-validation-split)
18. [Cell 36 – TF-IDF Vectorization](#cell-36--tf-idf-vectorization)
19. [Cell 37 – Scaling & Combining](#cell-37--scaling--combining)
20. [Cells 38–41 – Baseline Model](#cells-3841--baseline-model)
21. [Cell 42 – Cross-Validation](#cell-42--cross-validation)
22. [Cell 43 – Submission Generation](#cell-43--submission-generation)

---

## Cell 0 – Imports

### What's happening
We import all required libraries at the top. This is standard practice so failures surface immediately and the reader knows every dependency upfront.

### Library-by-library rationale

| Library | Why |
|---|---|
| `numpy` | Foundation for numerical operations; faster than Python loops |
| `pandas` | Tabular data manipulation, groupby, merges — the backbone of any EDA |
| `matplotlib` | Base plotting library; gives fine-grained control over figures |
| `seaborn` | Higher-level statistical plots built on matplotlib — less code, prettier defaults |
| `scipy.stats` | Statistical tests (Kruskal-Wallis, normality, Q-Q) — needed for hypothesis testing in EDA |
| `wordcloud` | Visual representation of text frequency — useful for quick text comprehension |
| `sklearn.*` | All ML utilities — split, scale, vectorise, model, evaluate |
| `re` | Regular expressions for text cleaning |
| `warnings` | Suppress deprecation noise that clutters output |

### Why `warnings.filterwarnings('ignore')`?
During EDA we get a lot of convergence and deprecation warnings that aren't actionable. We suppress them for cleaner output. In production code we'd handle these properly.

### Why `sns.set_style('whitegrid')`?
A grid background helps readers read values from plots. The "whitegrid" style is clean and professional — avoids the default grey background that looks amateur.

---

## Cell 1 – Data Loading

### What's happening
```python
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
```

We load three files:
- **train.csv** — has features + target (`label`)
- **test.csv** — has features only (no target) — we predict on this
- **Sample.csv** — shows the expected submission format

### Why print shapes?
To immediately verify we loaded the right files. If `train.shape` shows 0 rows, something went wrong. This is the first sanity check.

### Decision: Why not combine train + test for EDA?
We keep them separate because:
1. We never want to accidentally "peek" at test data distribution during analysis
2. Later preprocessing (scalers, encoders) must be fit on train ONLY
3. Combining would make it easy to introduce data leakage

---

## Cells 2–5 – Initial Inspection

### `train.sample(5, random_state=42)` vs `train.head(5)`
**`head()` always shows the first 5 rows.** If the data is sorted (e.g., by label), you get a biased view. `sample()` picks random rows, giving a more honest snapshot.

### `train.info()`
Shows:
- Column names and dtypes (is `upvote` stored as int or object?)
- Non-null counts (quick missing-value check)
- Memory usage (important for large datasets)

### `train.describe()` — What to look for
- **count < total rows?** → Missing values in that column
- **mean ≫ median?** → Right-skewed distribution or outliers
- **min = 0, 25% = 0?** → Many zeros (sparse feature)
- **std ≫ mean?** → Very high variance, might need scaling

### `train.describe(include='all')`
Adds categorical columns. Key stats:
- `unique` — how many distinct values
- `top` — most frequent value
- `freq` — how often `top` appears

### Separating numeric vs object columns
```python
num_cols = train.select_dtypes(include='number').columns.tolist()
cat_cols = train.select_dtypes(include='object').columns.tolist()
```
**Decision:** We do this early because different column types need completely different treatment:
- Numeric → distributions, correlations, scaling
- Object → value counts, encoding, text processing

---

## Cells 6–7 – Missing Value Audit

### The `missing_report()` function
Custom function that:
1. Counts nulls per column
2. Computes % missing
3. Filters to only show columns WITH missingness
4. Sorts by severity

**Why a function?** Because we apply the same check to both train and test.

### Missing value heatmap
```python
sns.heatmap(df.isnull(), ...)
```
Yellow cells = missing. This reveals **patterns**:
- Are the same rows missing across multiple columns? (systematic — MAR/MNAR)
- Random scattered yellow? (likely MCAR)

**Examiner Q: "Why does the mechanism matter?"**
→ MCAR: safe to drop or simple impute. MAR: need model-based imputation. MNAR: dangerous — the missingness itself carries information.

---

## Cell 8 – Duplicate Detection

### Why check duplicates?
Duplicate rows inflate certain classes, bias model training, and leak information (if the same comment appears in both train and test splits).

### Exact vs partial duplicates
```python
train.duplicated().sum()                      # exact — all columns match
train.duplicated(subset=['comment']).sum()     # partial — same text, different metadata
```
**Decision:** We only drop *exact* duplicates. Partial duplicates (same text, different metadata) might be legitimate reprosts.

---

## Cell 9 – Target Variable Analysis

### Why this matters most
The target `label` has 4 classes. The competition metric is **Macro F1** — this equally weighs F1 for ALL classes. If one class has only 2% of samples, a model that ignores it gets 0 F1 for that class, dragging down the macro average.

### Imbalance ratio
```python
imbalance_ratio = majority / minority
```
- Ratio ~1 → balanced → standard approach is fine
- Ratio >3 → imbalanced → need stratification + possibly class weights/SMOTE

### Why both bar chart AND pie chart?
- Bar chart → easy to compare absolute counts
- Pie chart → easy to see proportional share

### Stratified split decision
If imbalance exists, we use `stratify=y` in `train_test_split` to ensure each fold has the same label proportions as the full dataset.

---

## Cells 10–12 – Univariate: Numeric Features

### Separating binary vs continuous
```python
if unique_vals <= 2:
    binary_cols.append(col)
else:
    continuous_cols.append(col)
```
**Why?** Binary columns (like `race`, `gender` — 0/1 indicators) should be shown as bar charts, not histograms. A histogram of 0s and 1s is misleading.

### The triple-plot: Histogram + Box + Q-Q
For each continuous feature we plot three views:

1. **Histogram + KDE** — shows the distribution shape. KDE smooths it. The mean (red) vs median (green) lines reveal skewness:
   - Mean > Median → right-skewed
   - Mean ≈ Median → symmetric

2. **Box Plot** — shows quartiles + outliers. Points beyond the whiskers (1.5×IQR) are potential outliers.

3. **Q-Q Plot** — compares the data against a theoretical normal distribution. If points follow the diagonal line → approximately normal. Deviations at the tails → heavy/light tails.

### Skewness & Kurtosis table
- **Skewness > 1** → strongly right-skewed → consider log transform
- **Kurtosis > 3** → heavy tails → lots of outliers → RobustScaler preferred

**Examiner Q: "Why not just look at histograms?"**
→ Histograms are subjective (bin width changes perception). Skewness gives a single number; Q-Q plots are a more rigorous visual test.

---

## Cells 13–15 – Univariate: Text Feature

### Comment length vs word count
```python
train['comment_len']   = train['comment'].astype(str).apply(len)
train['comment_words'] = train['comment'].astype(str).apply(lambda x: len(x.split()))
```
These are cheap but effective features:
- Toxic comments might be shorter/longer than neutral ones
- Certain categories may have characteristic verbosity

### Word Cloud
A word cloud sizes each word by frequency. This is NOT rigorous — it's a visual tool for **generating hypotheses**:
- "Hmm, the word 'kill' appears large in label 2 but not label 0 — text content matters"
- This motivates using TF-IDF later

### Per-label word clouds
Comparing clouds across labels visually confirms whether different categories have distinct vocabulary. If they all look the same → text alone won't be enough.

---

## Cells 16–18 – Datetime Exploration

### Parsing with `errors='coerce'`
```python
pd.to_datetime(train['created_date'], errors='coerce')
```
`errors='coerce'` converts unparseable dates to NaT instead of crashing. Safe defensive coding.

### Extracted features
| Feature | Rationale |
|---|---|
| `hour` | Posting time might correlate with toxicity (late-night comments) |
| `day_of_week` | Weekday vs weekend behaviour differs |
| `month` | Seasonal patterns, event-driven spikes |
| `is_weekend` | Binary flag — simpler signal for models |

### Stacked bar by hour
Shows whether certain label categories peak at specific hours. If we see label 3 spiking at 2 AM, that's a date/time signal the model can exploit.

---

## Cells 19–22 – Bivariate Analysis

### Grouped box plots (numeric vs label)
```python
sns.boxplot(x='label', y=col, data=train)
```
Shows how each numeric feature distributes across label categories. If box positions are clearly separated → that feature is discriminative.

### Comment length vs label
If toxic comments are systematically shorter/longer, then `comment_len` is informative. We verify with:

### Kruskal-Wallis test
```python
stat, p = stats.kruskal(*groups)
```
**Why Kruskal-Wallis, not ANOVA?**
- ANOVA assumes normality + equal variances
- Kruskal-Wallis is non-parametric — works on any distribution
- Our features are clearly not normal (we saw skewness earlier)

If p < 0.05 → statistically significant difference across groups → feature is useful.

### Spearman correlation heatmap
```python
corr = df.corr(method='spearman')
```
**Why Spearman, not Pearson?**
- Pearson measures LINEAR correlation only
- Spearman measures MONOTONIC (any direction, not just straight-line) — more general
- Spearman uses ranks → robust to outliers

### Flagging high-correlation pairs (|r| > 0.8)
If two features are >0.8 correlated, they carry redundant information:
- Linear models get confused (multicollinearity)
- We might drop one later, or let regularisation handle it

### Binary features — stacked proportions
```python
ct = pd.crosstab(train[col], train['label'], normalize='index') * 100
```
For a binary col like `race`:
- If `race=1` has 50% label_2 but `race=0` has only 10% label_2 → race is discriminative
- `normalize='index'` shows row-wise percentages to see proportions, not raw counts

---

## Cell 23 – Cramér's V

### What is Cramér's V?
Pearson/Spearman work for numeric pairs. For categorical↔categorical pairs, we need **Cramér's V**:
```
V = √(χ² / (n × (k−1)))
```
- Range: [0, 1]
- 0 → no association
- 1 → perfect association

### Why build an association matrix?
To see which binary/indicator features are associated with each other AND with the target label. Strong V with label → that feature is predictive.

---

## Cell 24 – Multivariate / Pair Plot

### `sns.pairplot()`
Plots every numeric feature against every other, colored by label. This reveals:
- Which pairs of features together separate labels well (clusters in 2D)
- Non-linear relationships (curved patterns)

**Decision: Limit to 5 features.** Pair plots are O(n²) in compute and visual space. With 20 features we'd get 400 subplots — unreadable.

---

## Cells 25–26 – Outlier Detection

### IQR method
```
Lower fence = Q1 − 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```
Standard Tukey fences. For a normal distribution, only ~0.7% of data falls outside.

### Why not remove outliers?
**Decision:** We keep outliers for the baseline because:
1. Logistic Regression uses L2 regularisation, which is somewhat robust
2. Removing outliers loses information — those extreme cases might be the most important
3. Tree-based models (coming in Milestone 3–4) are completely outlier-robust
4. We can always revisit this decision after seeing baseline performance

### Box + strip overlay
The strip plot shows individual data points overlaid on the box. This exposes the density of outliers — if there are 5 dots vs 500 dots beyond the whisker, our treatment strategy differs.

---

## Cell 27 – Post-ID Analysis

### The problem with post_id
Post IDs have high cardinality (many unique values). One-hot encoding would create thousands of sparse columns → curse of dimensionality.

### Frequency encoding — the solution
```python
post_freq_train = train['post_id'].value_counts(normalize=True)
train['post_id_freq'] = train['post_id'].map(post_freq_train)
```
Replace each post_id with its frequency in the training set. Rationale:
- Popular posts (high freq) might be more controversial → more extreme labels
- It compresses the information into a single float column
- No risk of multicollinearity (unlike OHE)

### Why `.fillna(0)` for test?
Some post_ids in test may not appear in train → unseen IDs get frequency 0.

---

## Cells 28–33 – Feature Engineering

### Text cleaning — `clean_text()`

| Step | Code | Why |
|---|---|---|
| Lowercase | `text.lower()` | "HELLO" and "hello" are the same word |
| Remove URLs | `re.sub(r'http\S+', ...)` | URLs are noise — they don't generalise |
| Remove @mentions | `re.sub(r'@\w+', ...)` | Usernames are unique identifiers, not content |
| Replace digits | `re.sub(r'\d+', ' NUM ')` | The specific number rarely matters; the PRESENCE of numbers might |
| Collapse whitespace | `re.sub(r'\s+', ' ')` | Clean up formatting artifacts |

**Decision: NOT removing punctuation.** TF-IDF with n-grams can capture patterns like "!!!" or "???" which signal emotion.

### Engagement features
```python
train['vote_ratio'] = train['upvote'] / (train['total_votes'] + 1)
```
- **+1 in denominator:** Avoids division by zero when total_votes = 0
- **vote_ratio:** Captures sentiment polarity independent of total engagement
- **vote_diff:** Captures absolute sentiment direction

### Identity flag sum & any_identity
```python
train['identity_flag_sum'] = train[identity_cols].sum(axis=1)
train['any_identity'] = (train['identity_flag_sum'] > 0).astype(int)
```
- `identity_flag_sum`: How many identity groups are mentioned (intensity)
- `any_identity`: Binary — does the comment mention ANY identity group?

These aggregate features give the model both granular and coarse signals.

### Uppercase ratio
```python
sum(1 for c in x if c.isupper()) / (len(x) + 1)
```
"STOP DOING THIS" has high uppercase ratio → signals shouting/anger. Toxic comments might have higher uppercase usage.

### Exclamation and question counts
Simple but effective:
- Many `!` → strong emotion
- Many `?` → questioning, sarcasm

---

## Cell 34 – Feature Matrix Preparation

### Why create a `structured_features` list?
Explicit feature selection has three benefits:
1. **Auditability** — anyone can see exactly what goes into the model
2. **Reproducibility** — adding/removing a feature is a one-line change
3. **Safety** — we don't accidentally include the target or leak columns

### Safety check
```python
structured_features = [c for c in structured_features if c in train.columns and c in test.columns]
```
If a column doesn't exist in both train and test, it would crash during prediction. This guard prevents that.

---

## Cell 35 – Train-Validation Split

### Why 80/20?
Standard ML practice. 80% gives enough training data; 20% gives a statistically meaningful validation set. For our dataset size, this is appropriate.

### `stratify=y` — Critical decision
Without stratification, random chance might give our validation set a completely different label distribution than training. With stratification, each split mirrors the original proportions.

**Examiner Q: "What if you didn't stratify?"**
→ If minority class makes up 5% of data, a random 20% split might not include enough minority samples in validation. Macro F1 would be unreliable.

### `random_state=42`
Ensures reproducibility. Anyone running the same code gets the same split.

### Printing distributions
We verify that stratification worked by printing label proportions in both splits.

---

## Cell 36 – TF-IDF Vectorization

### What is TF-IDF?
**TF (Term Frequency):** How often a word appears in a document.
**IDF (Inverse Document Frequency):** How rare the word is across all documents.

TF × IDF = High score for words that are frequent in THIS document but rare overall. "the" gets low score (common everywhere). "xenophobic" gets high score (rare but meaningful).

### Hyperparameter choices

| Parameter | Value | Rationale |
|---|---|---|
| `max_features=15000` | Limits vocabulary to top 15k terms. Prevents memory explosion while keeping most informative words |
| `ngram_range=(1,2)` | Captures single words AND two-word phrases ("hate speech", "very good"). Trigrams add too much noise at this stage |
| `sublinear_tf=True` | Uses `log(1+tf)` instead of raw count. A word appearing 100x isn't 100x more important than one appearing once — diminishing returns |
| `min_df=3` | Ignores terms in fewer than 3 documents. Extremely rare words are noise (typos, proper nouns) |
| `max_df=0.95` | Ignores terms in >95% of documents. These are essentially stop words the built-in list missed |
| `stop_words='english'` | Removes common English words (the, is, at). They carry no discriminative signal |

### fit on train ONLY
```python
X_train_tfidf = tfidf.fit_transform(X_train_split['comment_clean'])
X_val_tfidf   = tfidf.transform(X_val_split['comment_clean'])
```
**This is the single most important anti-leakage rule.** If we fit TF-IDF on train+val:
- IDF scores include validation data → the model "knows" test word frequencies
- This inflates validation scores → false confidence

---

## Cell 37 – Scaling & Combining

### Why `SimpleImputer(strategy='median')` before scaling?
StandardScaler crashes on NaN. We impute first.

**Why median, not mean?** Median is robust to outliers. If `upvote` has a few comments with 10,000 upvotes, the mean would be inflated, but median stays stable.

### Why StandardScaler for structured features?
- Logistic Regression uses gradient-based optimisation
- Features on different scales (upvotes: 0–10000, is_weekend: 0–1) cause the gradient to oscillate
- Scaling puts everything on the same scale → faster convergence

### Combining with `hstack`
```python
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_struct)])
```
TF-IDF produces a **sparse matrix** (mostly zeros). We convert structured features to sparse too, then horizontally stack them. This is memory-efficient — a dense 15000-column matrix would eat gigabytes.

---

## Cells 38–41 – Baseline Model

### Why Logistic Regression as baseline?

| Property | Benefit |
|---|---|
| Fast to train | Can iterate quickly even with 15k TF-IDF features |
| Probabilistic output | `predict_proba()` useful for later ensembling/blending |
| Interpretable | Coefficients tell us which features drive each class |
| Regularised (L2) | Handles multicollinearity from TF-IDF naturally |
| Strong baseline | Often competitive — not "too simple" |

### Key hyperparameters

| Parameter | Value | Why |
|---|---|---|
| `C=1.0` | Default regularisation strength. Higher C → less regularisation → closer to unregularised model |
| `solver='lbfgs'` | Best solver for multiclass with L2 penalty. Faster than 'saga' when number of features > number of samples |
| `max_iter=1000` | Ensures convergence. Default 100 is often not enough for high-dimensional TF-IDF |
| `multi_class='multinomial'` | True multinomial (softmax). Alternative 'ovr' trains one-vs-rest independently — less principled for >2 classes |
| `random_state=42` | Reproducibility |
| `n_jobs=-1` | Use all CPU cores for any parallel steps |

### Macro F1 — the competition metric
```python
macro_f1 = f1_score(y_val_split, y_val_pred, average='macro')
```
**Macro F1** = average of per-class F1 scores, unweighted. Every class matters equally, regardless of size:
```
Macro F1 = (F1_class0 + F1_class1 + F1_class2 + F1_class3) / 4
```

**Why not accuracy?** With 80/10/5/5 class splits, a model that always predicts the majority class gets 80% accuracy but 0.25 macro F1. The metric forces us to care about minority classes.

### Confusion matrix
Rows = actual labels, columns = predicted labels. Diagonal = correct predictions. Off-diagonal cells show *where* the model confuses classes. This tells us which classes are hardest to distinguish → guides future feature engineering.

### Per-class F1 breakdown
Shows which specific class the model struggles with. If one class has F1 = 0.3 while others are 0.8, that's the bottleneck dragging down macro F1.

---

## Cell 42 – Cross-Validation

### Why 5-Fold Stratified CV?
A single train/val split is noisy — the val score depends on which 20% of data was held out. 5-fold CV:
1. Splits data into 5 parts
2. Trains on 4 parts, validates on 1 part
3. Repeats 5 times (each part is the validation set once)
4. Reports mean ± std of scores

**Stratified** ensures each fold has the same label proportions.

### Interpreting the output
```
5-Fold CV Macro F1: 0.XXXX ± 0.YYYY
```
- Mean is our best estimate of true performance
- Std tells us stability: ±0.01 → stable; ±0.05 → noisy, need more data or better features
- If single-split val score is very different from CV mean → the original split was lucky/unlucky

### Why refit on full train for CV?
```python
X_full_tfidf = tfidf.fit_transform(X_train_full['comment_clean'])
```
CV handles its own internal splits. The TF-IDF and scaler here are fitted on all train data, then CV does its own splitting. This is *slightly* leaky for the TF-IDF (ideally TF-IDF would be inside a Pipeline inside CV), but it's a common pragmatic choice for baseline evaluation. We'll fix this with proper Pipelines in Milestone 2.

---

## Cell 43 – Submission Generation

### Workflow
1. Retrain final model on ALL training data (not 80% — we want every drop of signal)
2. Transform test set using TF-IDF/scaler fitted on all training data
3. Predict
4. Format as the sample submission requires
5. Save to CSV

### Distribution check
```python
pd.Series(test_predictions).value_counts().sort_index()
```
Sanity check: if the model predicts 99% class 0 for test → something is wrong. The distribution should roughly mirror the training distribution.

---

## Decisions & Design Philosophy — Quick Reference

| Decision | Alternatives Considered | Why This Choice |
|---|---|---|
| Spearman over Pearson | Pearson is more common | Our features are non-normal; Spearman handles monotonic relationships |
| Median imputation | Mean, KNN, MICE | Median is robust, fast, and sufficient for a baseline |
| TF-IDF over BoW | CountVectorizer | TF-IDF downweights common words automatically → better signal |
| StandardScaler over MinMax | MinMax, RobustScaler | LogReg likes zero-centered data; StandardScaler is the default for linear models |
| Keep outliers | Remove, clip, winsorise | LogReg with L2 is somewhat robust; removing loses signal; revisit later |
| Frequency encoding for post_id | OHE, target encoding | OHE too many columns; target encoding leaks; frequency is safe |
| `+1` in denominators | `epsilon=1e-9` | Cleaner semantically for count data |
| 15k TF-IDF features | 5k, 30k, 50k | 15k captures most vocabulary without excessive dimensionality |
| Split before TF-IDF fitting | Fit on all data first | Prevents test information leaking into IDF weights |

---

## Potential Examiner Questions & Suggested Answers

**Q: Why did you use `sublinear_tf=True` in TF-IDF?**
> Standard TF uses raw counts. But if a word appears 100 times vs 10 times, it's not 10x more important — there are diminishing returns. `sublinear_tf` applies `log(1 + tf)`, which compresses high frequencies. This is mathematically equivalent to saying we believe importance grows logarithmically with frequency.

**Q: Why Kruskal-Wallis and not a t-test or ANOVA?**
> Our numeric features are clearly non-normal (high skewness, heavy tails from the Q-Q plots). ANOVA assumes normality and equal variances. Kruskal-Wallis is non-parametric — it ranks the data and tests whether rank distributions differ across groups. It makes no distributional assumptions.

**Q: What's the curse of dimensionality and how does it affect your TF-IDF features?**
> With 15,000 TF-IDF features and, say, 100,000 training samples, we're in a moderately high-dimensional space. The curse says that as dimensions grow, all points become approximately equidistant. This particularly hurts KNN and kernel SVM. Logistic Regression handles it better because L2 regularisation shrinks coefficients — it effectively ignores uninformative features.

**Q: Why did you choose Macro F1 and not accuracy?**
> Macro F1 treats all classes equally. If the competition used accuracy, a model that always predicted the majority class would score high despite being useless. Macro F1 penalises poor performance on ANY class, which aligns with the goal of correctly categorising ALL types of comments.

**Q: Explain the `drop='first'` in OneHotEncoder — why isn't it used here?**
> We didn't use OneHotEncoder because our categorical features are mostly binary (0/1 indicators) or high-cardinality (post_id, which we frequency-encoded). `drop='first'` removes one dummy column to avoid multicollinearity in linear models. We mentioned it for completeness — it becomes relevant in later milestones if we introduce ordinal features.

**Q: Why not do PCA at this stage?**
> PCA is a dimensionality reduction technique — useful if we have too many features. With TF-IDF + 25 structured features, the Logistic Regression with L2 regularisation handles the dimensionality fine. PCA destroys feature interpretability (PC1 is a mix of all features), and for a baseline we want to understand WHICH features matter. PCA comes in Milestone 3.

**Q: How would you handle the data differently if the imbalance ratio was 50:1?**
> With extreme imbalance I'd consider: (1) SMOTE (synthetic oversampling of minority class), (2) class_weight='balanced' in the model, (3) threshold tuning on predicted probabilities, (4) stratified sampling with more folds. For this dataset, stratification alone is sufficient.

---

## Key Takeaways for Milestone 2
1. The baseline gives us a reference Macro F1 score to beat
2. Text (TF-IDF) is likely the strongest signal — tune TF-IDF params
3. Engagement features (votes) and identity flags show bivariate signal
4. Next steps: SGD classifier, hyperparameter tuning, class weights, better TF-IDF configs
