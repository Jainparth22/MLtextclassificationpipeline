# 🎓 Viva Preparation Guide — Comment Category Prediction Project

## Dataset Overview
- **Task:** Multi-class text classification (4 labels: 0, 1, 2, 3)
- **Train:** 197,999 rows × 15 columns | **Test:** 102,000 rows × 14 columns
- **Columns:** `comment`, `created_date`, `upvote`, `downvote`, `emoticon_1/2/3`, `if_1/2`, `race`, `religion`, `gender`, `disability`, `label`
- **Best Kaggle Score:** 0.826 (Macro F1)

---

## 1. EDA (Exploratory Data Analysis)

### What we did:
- **Class distribution** → Severe imbalance: Label 0 (57.7%), Label 2 (31.6%), Label 1 (8.0%), Label 3 (2.7%)
- **Missing values analysis** → `race`, `religion`, `gender` have ~75-85% missing; `comment` has few nulls
- **Comment length analysis** → Toxic comments (Label 3) tend to be longer, more caps
- **Correlation analysis** → `upvote`/`downvote` weakly correlated with labels
- **Feature distributions** → `upvote`, `downvote`, `if_1`, `if_2` are heavily right-skewed
- **Count plots** → Showed the severe 4-class imbalance visually

### A. Visualization Techniques

| Plot | What it shows | For our project |
|---|---|---|
| **Count plot** | Class frequencies | Shows 57.7% / 31.6% / 8% / 2.7% imbalance |
| **Violin plot** | Distribution shape + density per class | Comment length per label — Label 3 has longer, more spread-out comments |
| **Box plot** | Median, quartiles, outliers | Upvote/downvote distributions per label |
| **Histogram** | Frequency distribution | Shows right-skew in votes → justifies log1p transform |
| **Heatmap** | Correlation matrix | Feature-to-feature and feature-to-label relationships |
| **Pair plot** | Feature-vs-feature scatter | Relationships between numeric features by class |
| **Word cloud** | Most frequent words per class | Toxic words dominate in Label 1/3 |
| **Bar chart** | Top TF-IDF features per class | Which words are most discriminative |

**Violin plot example (viva-ready):**
```python
import seaborn as sns
# Violin plot shows FULL distribution shape — better than box plot
sns.violinplot(x="label", y="comment_length", data=train, palette="viridis")
plt.title("Comment Length Distribution by Label")
# Reveals: Label 3 has bimodal distribution (short insults + long rants)
```

### B. Correlation Analysis

#### Types of correlation:
| Method | When to use | Our project |
|---|---|---|
| **Pearson correlation** | Linear relationship, continuous features | `upvote` vs `downvote` (r ≈ 0.3 weak positive) |
| **Spearman correlation** | Monotonic relationship, ordinal/skewed data | Better for our skewed vote data |
| **Point-biserial** | Continuous vs binary | `comment_length` vs `disability` |
| **Cramér's V** | Two categorical variables | `race` vs `label` |

#### Our correlation findings:
```python
# Pearson correlation heatmap
corr = train[numeric_features + ["label"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
```

**Key correlations found:**
| Feature pair | Correlation | Interpretation |
|---|---|---|
| `upvote` ↔ `downvote` | ~0.3 (weak +) | Controversial comments get both |
| `upvote` ↔ `total_votes` | ~0.9 (strong +) | Expected — derived feature |
| `comment_length` ↔ `word_count` | ~0.95 (strong +) | Nearly perfect — could drop one |
| `comment_length` ↔ `label` | ~0.05 (negligible) | Length alone doesn't predict toxicity |
| `downvote` ↔ `label` | ~0.08 (very weak) | Slight tendency: toxic → more downvotes |

> **Viva answer:** "Pearson correlation showed most numeric features have weak correlation with labels (< 0.1), confirming that text content (TF-IDF) is the dominant signal. High inter-correlation between `comment_length` and `word_count` (r=0.95) means they're nearly redundant, but keeping both doesn't hurt linear models."

### C. Statistical Tests

| Test | What it tests | Our project example | Result |
|---|---|---|---|
| **Z-test** | Sample mean vs population mean (large n, known σ) | Is Label 3's mean `comment_length` significantly different from overall mean? | p < 0.001 → **Yes, significant** |
| **T-test** | Compare means of 2 groups (unknown σ) | Mean `downvotes` of Label 0 vs Label 3 | p < 0.05 → toxic comments get more downvotes |
| **Chi-squared test** | Independence of categorical variables | Is `has_race` independent of `label`? | p < 0.001 → **Not independent** (race flagging correlates with label) |
| **ANOVA** | Compare means across 3+ groups | Does `comment_length` differ across all 4 labels? | p < 0.001 → **Yes, at least one label differs** |
| **Kruskal-Wallis** | Non-parametric ANOVA (no normality assumption) | Same as ANOVA but for our skewed data | Better choice since votes are non-normal |
| **Shapiro-Wilk** | Tests normality | Is `upvote` normally distributed? | p < 0.001 → **Not normal** → justifies log1p |
| **Mann-Whitney U** | Non-parametric 2-group comparison | Compare `vote_ratio` between Label 0 vs Label 3 | Non-parametric alternative to t-test |

**Code examples (viva-ready):**
```python
from scipy import stats
from statsmodels.stats.weightstats import ztest

# Z-test: Is Label 3 comment length different from overall mean?
toxic = train[train["label"]==3]["comment_length"]
z_stat, p_val = ztest(toxic, value=train["comment_length"].mean())
print(f"Z={z_stat:.2f}, p={p_val:.6f}")  # p < 0.05 → significant

# Chi-squared: Is race flagging independent of label?
contingency = pd.crosstab(train["has_detection"], train["label"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi²={chi2:.2f}, p={p:.6f}")  # p < 0.001 → dependent

# ANOVA: Does comment_length differ across labels?
groups = [train[train["label"]==i]["comment_length"] for i in range(4)]
f_stat, p_val = stats.f_oneway(*groups)
print(f"F={f_stat:.2f}, p={p_val:.6f}")  # p < 0.001 → significant

# Shapiro-Wilk: Is upvote normally distributed?
stat, p = stats.shapiro(train["upvote"].sample(5000))
print(f"W={stat:.4f}, p={p:.6f}")  # p < 0.001 → NOT normal → log1p justified

# Kruskal-Wallis (non-parametric ANOVA — better for skewed data):
h_stat, p_val = stats.kruskal(*groups)
print(f"H={h_stat:.2f}, p={p_val:.6f}")
```

> **Viva answer:** "We used Shapiro-Wilk to confirm our numeric features are not normally distributed (p < 0.001), which justified our log1p transformation. ANOVA/Kruskal-Wallis confirmed that comment length and voting patterns differ significantly across labels, but with small effect sizes, meaning text content is still the primary signal. Chi-squared showed that demographic flags (race/religion/gender) are NOT independent of labels — toxicity detection is correlated with demographic mentions."

---

## 2. Data Preprocessing

### A. Missing Value Handling

#### What we did:
| Column | Method | Why |
|---|---|---|
| `comment` (text) | `fillna("")` + `dropna()` for nulls | Empty string = no text signal for TF-IDF |
| `race`, `religion`, `gender` | `fillna("missing")` | Preserves "missingness" as a category — null itself is informative (75-85% are null) |
| Numeric columns | No nulls found | `upvote`, `downvote` etc. have no missing values |

#### All possible methods for handling missing values:
| Method | How it works | When to use | Our case? |
|---|---|---|---|
| **Drop rows** | Remove rows with nulls | When few rows affected | ✅ Used for null comments (very few) |
| **Fill with constant** | `fillna("missing")` or `fillna(0)` | When nullness itself is meaningful | ✅ Used for race/religion/gender |
| **Fill with mean/median** | Replace with column average | For numeric columns with random missingness | ❌ Not needed (no numeric nulls) |
| **Fill with mode** | Replace with most frequent value | For categorical columns | ❌ Could use, but "missing" is more informative |
| **Forward/backward fill** | Use previous/next row's value | For time-series data | ❌ Not applicable (not time-series) |
| **KNN Imputation** | Predict missing from K nearest neighbors | Complex patterns in missingness | ❌ Too slow for 200K rows |
| **Iterative Imputation** | Model-based (like MICE) | When features are inter-dependent | ❌ Overkill for our data |
| **Indicator variable** | Add `has_X` binary column | When missingness is a signal | ✅ Used `has_detection`, `has_race`, etc. |

> **Viva answer:** "We used constant fill (`'missing'`) for categorical features because 75-85% are null — the absence of a race/religion/gender flag IS the signal. We also created binary indicator variables (`has_detection`) to let models learn from the missingness pattern itself."

---

### B. Text Preprocessing

#### What we did:
```
1. Lowercase                    → normalize case
2. Remove URLs                  → noise removal
3. Remove HTML tags             → noise removal
4. Remove emails                → noise removal
5. Remove standalone numbers    → reduce vocabulary
6. Keep only letters + apostrophes → clean for TF-IDF
7. Compress whitespace          → normalize spacing
```

#### What we tried and results:
| Technique | Result | Used in final? |
|---|---|---|
| Minimal cleaning (lowercase + remove noise) | **Best — 0.826 Kaggle** | ✅ Yes |
| NLTK Lemmatization | Same as no lemma (0.826) | ❌ Illegal library |
| Regex stemming | **Hurt scores** — over-stemmed ("tribal"→"trib") | ❌ No |
| Stop word removal | No stop_words slightly better | ✅ No stop_words |

#### All possible text preprocessing methods:
| Method | What it does | Impact in our case |
|---|---|---|
| **Lowercasing** | "STUPID" → "stupid" | ✅ Essential — normalizes vocabulary |
| **URL/HTML removal** | Remove noise tokens | ✅ Essential — reduces noise |
| **Stemming** (Porter/Snowball) | "running" → "run" | ❌ **Hurt scores** — too aggressive |
| **Lemmatization** (WordNet) | "running" → "run" (smarter) | ≈ **No effect** (also illegal) |
| **Stop word removal** | Remove "the", "is", "a" | ❌ **Slightly hurt** — stop words carry context |
| **Spell correction** | Fix typos | ❌ Too slow for 200K comments |
| **Contraction expansion** | "don't" → "do not" | ❌ Not tried, low expected impact |
| **Repeated char collapse** | "stuuuupid" → "stupid" | ❌ Tried in V6, negligible impact |

> **Viva answer:** "We found that minimal text cleaning outperformed aggressive preprocessing. TF-IDF with trigrams already captures morphological patterns, so stemming/lemmatization adds nothing. Stop word removal actually hurt because words like 'not' and 'you' carry sentiment context."

---

### C. Feature Engineering

#### What we created:

| Feature | Type | How |
|---|---|---|
| `hour`, `day_of_week`, `month` | Temporal | Extracted from `created_date` |
| `comment_length` | Text stat | `len(comment)` |
| `word_count` | Text stat | `len(comment.split())` |
| `avg_word_length` | Text stat | `comment_length / word_count` |
| `total_votes` | Engagement | `upvote + downvote` |
| `vote_diff` | Sentiment | `upvote - downvote` |
| `vote_ratio` | Sentiment | `upvote / (downvote + 1)` |
| `total_emoticons` | Behavior | `emoticon_1 + emoticon_2 + emoticon_3` |
| `has_detection` | Demographic | Any race/religion/gender flagged |

#### Additional features tried in V6 (negligible impact):
| Feature | Expected signal | Actual impact |
|---|---|---|
| `caps_ratio` | Shouting = toxic | Negligible |
| `toxic_word_count` | Direct profanity count | Negligible |
| `excl_count`, `quest_count` | Emotional punctuation | Negligible |
| One-hot demographics | Race/religion/gender categories | Negligible |

> **Viva answer:** "TF-IDF dominates feature importance. Hand-crafted features add minimal signal because TF-IDF trigrams already capture patterns like 'you are stupid' or exclamation usage. The numeric features (votes, length) provide slight complementary signal."

---

### D. Feature Transformation

#### What we did:
| Transform | Applied to | Why |
|---|---|---|
| **Log1p** (`np.log1p`) | Skewed numerics (upvote, downvote, etc.) | Reduces skewness, helps linear models |
| **StandardScaler** | All numeric features | Zero mean, unit variance — required for SVM/LR |
| **TF-IDF** (sublinear) | Comment text | Converts text → sparse numeric vectors |

#### All possible transformations:
| Method | What it does | Our case |
|---|---|---|
| **StandardScaler** | Zero mean, unit variance | ✅ Used for numeric |
| **MinMaxScaler** | Scale to [0, 1] | ❌ Not needed (StandardScaler works for SVM) |
| **MaxAbsScaler** | Scale to [-1, 1] by max abs | ❌ Alternative, not tried |
| **Log1p transform** | Reduce right-skew | ✅ Used for votes, lengths |
| **Box-Cox / Yeo-Johnson** | Power transform for normality | ❌ Overkill for our linear models |
| **OneHotEncoder** | Categorical → binary columns | ✅ Tried in V6 (negligible impact) |
| **LabelEncoder** | Categories → integers | ❌ Creates false ordinal relationship |

---

## 3. Feature Selection / Extraction

### TF-IDF Configuration (Most Critical Decision)

| Parameter | Our choice | Why |
|---|---|---|
| **Word n-grams** | (1, 3) — unigrams + bigrams + trigrams | Captures "you are stupid" as single feature |
| **Char n-grams** | (3, 5) — character-level | Captures subword patterns, handles misspellings |
| **max_features** | 35K word + 25K char = 60K | Balance between signal and memory/speed |
| **sublinear_tf** | True | `1 + log(tf)` — reduces impact of very frequent terms |
| **min_df** | 3 | Ignore words appearing < 3 times |
| **max_df** | 0.9 | Ignore words appearing in > 90% of docs |
| **stop_words** | None (no removal) | Removing stop words slightly hurt (V2 experiment) |

### Feature Selection Methods:

| Method | Status | Result |
|---|---|---|
| **TF-IDF max_features** | ✅ Used | Controls vocabulary size (35K word + 25K char) |
| **Chi² SelectKBest** | 🔲 V7 experiment | Tests if reducing from 60K helps |
| **Mutual Information** | ❌ Not tried | Non-linear feature importance |
| **L1 regularization** (Lasso) | ❌ Not tried | Automatic sparse feature selection |
| **Variance Threshold** | ❌ Not tried | Remove near-zero variance features |
| **PCA / TruncatedSVD** | ✅ Tried in V4 | 200 dims captured only 18.1% variance — too lossy |

> **Viva answer:** "TF-IDF IS our feature selection. `max_features=35000` for word n-grams selects the 35K most informative terms. We also tried SVD dimensionality reduction but it lost 82% of variance. Chi² is being tested in V7."

---

## 4. Class Imbalance Handling

### The problem:
```
Label 0:  57.7%  (114,475 samples)  — majority
Label 2:  31.6%  (62,434 samples)   — large minority
Label 1:   8.0%  (15,918 samples)   — small minority
Label 3:   2.7%  (5,469 samples)    — RARE class ← main bottleneck
```

### What we tried:

| Method | How it works | Result |
|---|---|---|
| **class_weight="balanced"** | Adjusts loss function: `weight = n_samples / (n_classes × n_class_i)` | ✅ **Best for ensembles** (0.826 Kaggle) |
| **Custom weights {0:1,1:2,2:1,3:6}** | Manual per-class weights | ✅ **Best for standalone LinearSVC** (0.822 Kaggle) |
| **RandomOverSampler (ROS)** | Duplicates minority samples | ❌ **Hurt** — double correction with balanced weights |
| **SMOTE** | Generates synthetic minority samples | ❌ **Hurt** — synthetic 60K-dim samples added noise |
| **No handling** | Ignore imbalance | ❌ **Poor** — model ignores Label 3 entirely |

### All possible methods for class imbalance:
| Method | Category | Our case |
|---|---|---|
| **class_weight="balanced"** | Cost-sensitive | ✅ Best approach |
| **Custom class weights** | Cost-sensitive | ✅ Second-best |
| **RandomOverSampler** | Oversampling | ❌ Double correction with balanced weights |
| **SMOTE** | Synthetic oversampling | ❌ Noisy in high-dimensional sparse data |
| **ADASYN** | Adaptive synthetic | ❌ Same issue as SMOTE |
| **Random Undersampling** | Undersampling | ❌ Would discard 95% of Label 0 data |
| **Tomek Links** | Boundary cleaning | ❌ Not tried |
| **SMOTE + Tomek** | Combined | ❌ Not tried |
| **Threshold tuning** | Post-processing | ❌ **Tried — always worse** (0.811 vs 0.818) |
| **Focal Loss** | Modified loss | ❌ Not available in sklearn |

> **Viva answer:** "`class_weight='balanced'` was the best approach because it mathematically increases the loss contribution of minority classes without altering the data distribution. We tried SMOTE and ROS, but they HURT scores because: (1) combining oversampling with balanced weights double-corrects, and (2) generating synthetic samples in 60K-dimensional sparse TF-IDF space creates noisy, unrealistic samples."

---

## 5. Model Selection Journey (M1 → M5)

| Milestone | Best Model | Kaggle | Key Learning |
|---|---|---|---|
| M1 | Multinomial NB | ~0.65 | Baseline with basic TF-IDF |
| M2 | Logistic Regression | ~0.72 | Better than NB for multi-class |
| M3 | LinearSVC (C=0.1) | 0.816 | SVM dominates text classification |
| M4 | Stacking (4→LR) | 0.823 | Ensembles > individual models |
| M5 V1 | Stacking + lemma + 45K | 0.826 | More features + trigrams |
| M5 V3 | Stacking + 60K + hybrid | 0.826 | No stemming = same result |
| M5 V4 | ROS + SMOTE + LightGBM | OOM | All V4 changes hurt |
| M5 V6 | +ComplementNB +LightGBM meta | 0.787 | Adding weak models hurts |
| **M5 V5** | **Stacking (4→LR, 60K)** | **0.826** | **Simple > Complex** |

---

## 6. Key Viva Takeaways

1. **"Why not deep learning?"** → Competition rules only allow sklearn, xgboost, lightgbm. No PyTorch/TensorFlow/NLTK.

2. **"Why TF-IDF over Bag of Words?"** → TF-IDF weighs rare, discriminative words higher. `sublinear_tf=True` further reduces impact of very common terms.

3. **"Why character n-grams?"** → Captures subword patterns (prefixes/suffixes), robust to misspellings ("stuupid" still matches character patterns).

4. **"Why Stacking > individual models?"** → Meta-learner (LR) learns optimal weighted combination of 4 diverse base models' predictions. Better than any single model.

5. **"Why didn't SMOTE/oversampling help?"** → Two reasons: (a) `class_weight` already handles imbalance mathematically, (b) synthetic samples in 60K sparse dimensions are noisy and unrealistic.

6. **"What's the 0.826 ceiling?"** → TF-IDF treats text as bag-of-words — can't understand sarcasm, context, or implicit toxicity. Breaking 0.84+ requires transformer-based models (BERT etc.) which aren't allowed.
