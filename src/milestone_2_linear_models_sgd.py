# %% [markdown]
# # Milestone 2: Linear Models, SGD & Hyperparameter Tuning
# **Comment Category Prediction Challenge**

# **Milestone 1 Baseline:** LogReg + TF-IDF -> Accuracy: 0.8897 | Macro F1: 0.7770

# **This notebook covers:**
# 1. Advanced Text Preprocessing & NLP Analysis (M2 Questions)
# 2. Feature Engineering (temporal, emoticon-derived, text stats)
# 3. **Optimized Fast Pipeline** — TF-IDF precomputed once, sparse hstack
# 4. Feature Scaling Experiments (Standard vs MinMax vs Log+Standard)
# 5. **Model 1**: Tuned Logistic Regression (GridSearchCV)
# 6. **Model 2**: SGDClassifier with log_loss
# 7. **Model 3**: SGDClassifier with hinge loss (Linear SVM)
# 8. **Model 4**: RidgeClassifier (L2-regularized)
# 9. Model Comparison & Best Submission

# **Optimization:** TF-IDF is computed ONCE and reused across all models.
# This reduces runtime from ~2 hours to ~15-25 minutes.

# ---

# %% [markdown]
# ## 1. Setup & Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
from collections import Counter

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV,
                                     cross_val_score)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, make_scorer)

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")
sns.set_palette("viridis")

print("All libraries imported successfully.")

# %% [markdown]
# ## 2. Data Loading

# %%
train = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/train.csv")
test  = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/test.csv")
sample = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/Sample.csv")

print(f"Train: {train.shape} | Test: {test.shape} | Sample: {sample.shape}")

# %% [markdown]
# ---
# ## 3. Milestone 2 Questions — Text Preprocessing & NLP Analysis

# %%

# [Q] M2 Q1: In the context of text preprocessing for NLP,
# which condition must be checked in addition to NaN values when
# assessing missingness in the comment column of train.csv?

nan_count = train["comment"].isna().sum()
empty_string_count = (train["comment"].astype(str).str.strip() == "").sum()
whitespace_only = train["comment"].astype(str).str.match(r'^\s*$').sum()

print(f"NaN values in comment: {nan_count}")
print(f"Empty string comments: {empty_string_count}")
print(f"Whitespace-only comments: {whitespace_only}")
print(f"\n-> ANSWER: In addition to NaN, check for EMPTY STRINGS ('') and whitespace-only strings.")

# %% [markdown]
# **Answer:** Besides NaN values, we must check for **empty strings** and
# **whitespace-only strings** — they have no meaningful text content for NLP.

# %%
# Clean: Drop NaN comments
train = train.dropna(subset=["comment"])
print(f"After dropping NaN comments: {train.shape[0]:,} rows")

# %%

# [Q] M2 Q2: Convert created_date to datetime. Identify the month
# that occurs the most number of times in the dataset.

train["created_date"] = pd.to_datetime(train["created_date"], errors="coerce")
test["created_date"]  = pd.to_datetime(test["created_date"], errors="coerce")

train["month"] = train["created_date"].dt.month
month_counts = train["month"].value_counts().sort_values(ascending=False)

print("Comment counts by month:")
print(month_counts)
print(f"\n-> ANSWER: Most frequent month = Month {month_counts.index[0]} "
      f"({month_counts.values[0]:,} comments)")

# %%
plt.figure(figsize=(10, 5))
month_counts.sort_index().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Number of Comments by Month", fontsize=14, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%

# [Q] M2 Q3: Create total_emoticons = emoticon_1 + emoticon_2 + emoticon_3.
# What is the maximum value observed?

train["total_emoticons"] = train["emoticon_1"] + train["emoticon_2"] + train["emoticon_3"]
test["total_emoticons"]  = test["emoticon_1"] + test["emoticon_2"] + test["emoticon_3"]

max_emoticons = train["total_emoticons"].max()
print(f"Maximum total_emoticons: {max_emoticons}")
print(f"\n-> ANSWER: Maximum value of total_emoticons = {max_emoticons}")
print(f"\nDistribution:\n{train['total_emoticons'].value_counts().sort_index()}")

# %%

# [Q] M2 Q4: Calculate the median character length (including spaces)
# of comments where label == 3. (Missing comments = empty strings)

train["comment_filled"] = train["comment"].fillna("")
train["comment_char_length"] = train["comment_filled"].str.len()

median_len_label3 = train.loc[train["label"] == 3, "comment_char_length"].median()
print(f"Median character length for Label 3: {median_len_label3}")
print(f"\n-> ANSWER: Median character length for label=3 = {median_len_label3}")

# %%

# [Q] M2 Q5: Min-Max scale the upvote column to [0,1].
# What is the scaled value for upvote = 10?

up_min = train["upvote"].min()
up_max = train["upvote"].max()
scaled_10 = (10 - up_min) / (up_max - up_min)

print(f"Upvote range: min={up_min}, max={up_max}")
print(f"Scaled value for 10: (10 - {up_min}) / ({up_max} - {up_min}) = {scaled_10:.6f}")
print(f"\n-> ANSWER: Min-Max scaled value for upvote=10 = {scaled_10:.6f}")

# %%

# [Q] M2 Q6: Average word count (whitespace-separated) for label == 1?

train["word_count"] = train["comment"].astype(str).str.split().str.len()
avg_wc_label1 = train.loc[train["label"] == 1, "word_count"].mean()

print(f"Average word count for Label 1: {avg_wc_label1:.4f}")
print(f"\n-> ANSWER: Average word count for label=1 = {avg_wc_label1:.2f}")

# %%

# [Q] M2 Q7: How many comments contain "Trump"? (Case-insensitive)

trump_count = train["comment"].astype(str).str.contains("trump", case=False, na=False).sum()
print(f"Comments containing 'Trump': {trump_count}")
print(f"\n-> ANSWER: {trump_count} comments contain 'Trump' (case-insensitive)")

# %%

# [Q] M2 Q8: First comment (index 0): remove punctuation, remove
# stop words. How many words remain?

first_comment = train.iloc[0]["comment"]
print(f"Original comment (index 0):\n'{first_comment}'\n")
print(f"Original word count: {len(first_comment.split())}")

# Step 1: Remove all punctuation
comment_no_punct = first_comment.translate(str.maketrans("", "", string.punctuation))
print(f"\nAfter punctuation removal:\n'{comment_no_punct}'")
print(f"Word count: {len(comment_no_punct.split())}")

# Step 2: Remove stop words
words_after = comment_no_punct.lower().split()
words_remaining = [w for w in words_after if w not in ENGLISH_STOP_WORDS]

print(f"\nAfter stop word removal:")
print(f"Remaining words: {words_remaining}")
print(f"\n-> ANSWER: {len(words_remaining)} words remain after punctuation + stop word removal")

# %%

# [Q] M2 Q9: Lowercase + whitespace tokenize all comments.
# Total number of unique tokens?

all_tokens = train["comment"].astype(str).str.lower().str.split()
unique_tokens = set(token for tokens in all_tokens for token in tokens)

print(f"Total unique tokens: {len(unique_tokens):,}")
print(f"\n-> ANSWER: Total unique tokens = {len(unique_tokens):,}")

# %%

# [Q] M2 Q10: TfidfVectorizer with stop_words="english", min_df=5,
# ngram_range=(1,2). How many features?

tfidf_q = TfidfVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))
tfidf_q_matrix = tfidf_q.fit_transform(train["comment"].astype(str))

print(f"TF-IDF matrix shape: {tfidf_q_matrix.shape}")
print(f"\n-> ANSWER: TF-IDF features generated = {tfidf_q_matrix.shape[1]:,}")

del tfidf_q, tfidf_q_matrix  # Free memory

# %% [markdown]
# ---
# ## 4. Complete Feature Engineering

# %%
# Temporal features
for df in [train, test]:
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["hour"] = df["created_date"].dt.hour
    df["day_of_week"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.month

# Text-derived features
for df in [train, test]:
    df["comment"] = df["comment"].fillna("")
    df["comment_length"]  = df["comment"].astype(str).str.len()
    df["word_count"]      = df["comment"].astype(str).str.split().str.len().fillna(0).astype(int)
    df["avg_word_length"] = df["comment_length"] / (df["word_count"] + 1)

# Engagement-derived features
for df in [train, test]:
    df["total_votes"] = df["upvote"] + df["downvote"]
    df["vote_diff"]   = df["upvote"] - df["downvote"]
    df["vote_ratio"]  = df["upvote"] / (df["downvote"] + 1)

# Emoticon-derived (already created above for Q3, ensure test also has it)
for df in [train, test]:
    if "total_emoticons" not in df.columns:
        df["total_emoticons"] = df["emoticon_1"] + df["emoticon_2"] + df["emoticon_3"]

# Detection flag
for df in [train, test]:
    for col in ["race", "religion", "gender"]:
        df[col] = df[col].fillna("missing")
    df["has_detection"] = (df[["race", "religion", "gender"]].ne("missing").any(axis=1)).astype(int)

print("Feature engineering complete.")
print(f"Train: {train.shape} | Test: {test.shape}")

# %% [markdown]
# ---
# ## 5. Optimized Fast Pipeline — Precompute TF-IDF Once
#
# **Key optimization:** Instead of putting TF-IDF inside every sklearn Pipeline
# (which recomputes it for every CV fold), we compute TF-IDF **once** and
# reuse the sparse matrix across all models. This is 6-10× faster.

# %%
# --------------------------------------------------
# STEP 1: Compute TF-IDF ONCE
# --------------------------------------------------
print("Computing TF-IDF (one-time operation)...")
t0 = time.time()

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

X_text_train = tfidf.fit_transform(train["comment"].astype(str))
X_text_test  = tfidf.transform(test["comment"].astype(str))

print(f"TF-IDF done in {time.time()-t0:.1f}s")
print(f"   Train text matrix: {X_text_train.shape}")
print(f"   Test text matrix:  {X_text_test.shape}")

# %%
# --------------------------------------------------
# STEP 2: Prepare Numeric & Binary Features
# --------------------------------------------------

numeric_features = [
    "upvote", "downvote", "if_1", "if_2",
    "comment_length", "word_count", "avg_word_length",
    "total_votes", "vote_diff", "vote_ratio",
    "total_emoticons", "hour", "day_of_week", "month"
]

binary_features = [
    "emoticon_1", "emoticon_2", "emoticon_3",
    "disability", "has_detection"
]

X_num_train = train[numeric_features].values
X_bin_train = train[binary_features].values.astype(float)

X_num_test = test[numeric_features].values
X_bin_test = test[binary_features].values.astype(float)

y = train["label"].values

print(f"Numeric features: {len(numeric_features)} | Binary features: {len(binary_features)}")

# %% [markdown]
# ### 5.1 Feature Scaling Experiments
#
# From M1: numerical features are heavily right-skewed (upvote: 7.10, downvote: 10.06, if_1: 63.70).
# We compare three scaling strategies.

# %%
# --------------------------------------------------
# STEP 3A: StandardScaler
# --------------------------------------------------
scaler_std = StandardScaler()
X_num_std_train = scaler_std.fit_transform(X_num_train)
X_num_std_test  = scaler_std.transform(X_num_test)

# Combine: TF-IDF + Scaled Numerics + Binary -> single sparse matrix
X_all_std = hstack([X_text_train, csr_matrix(X_num_std_train), csr_matrix(X_bin_train)])
X_test_all_std = hstack([X_text_test, csr_matrix(X_num_std_test), csr_matrix(X_bin_test)])

print(f"StandardScaler -> Combined matrix: {X_all_std.shape}")

# %%
# --------------------------------------------------
# STEP 3B: MinMaxScaler
# --------------------------------------------------
scaler_mm = MinMaxScaler()
X_num_mm_train = scaler_mm.fit_transform(X_num_train)
X_num_mm_test  = scaler_mm.transform(X_num_test)

X_all_mm = hstack([X_text_train, csr_matrix(X_num_mm_train), csr_matrix(X_bin_train)])
X_test_all_mm = hstack([X_text_test, csr_matrix(X_num_mm_test), csr_matrix(X_bin_test)])

print(f"MinMaxScaler -> Combined matrix: {X_all_mm.shape}")

# %%
# --------------------------------------------------
# STEP 3C: Log1p Transform + StandardScaler
# --------------------------------------------------
skewed_cols = ["upvote", "downvote", "if_1", "if_2",
               "total_votes", "vote_ratio", "comment_length", "word_count"]
skewed_indices = [numeric_features.index(c) for c in skewed_cols]

X_num_log_train = X_num_train.copy()
X_num_log_test  = X_num_test.copy()

for idx in skewed_indices:
    X_num_log_train[:, idx] = np.log1p(np.clip(X_num_log_train[:, idx], 0, None))
    X_num_log_test[:, idx]  = np.log1p(np.clip(X_num_log_test[:, idx], 0, None))

print("Skewness comparison (selected features):")
for col, idx in zip(skewed_cols, skewed_indices):
    orig_skew = pd.Series(X_num_train[:, idx]).skew()
    log_skew  = pd.Series(X_num_log_train[:, idx]).skew()
    print(f"  {col:20s} -> Before: {orig_skew:8.2f}  |  After log1p: {log_skew:8.2f}")

scaler_log_std = StandardScaler()
X_num_logstd_train = scaler_log_std.fit_transform(X_num_log_train)
X_num_logstd_test  = scaler_log_std.transform(X_num_log_test)

X_all_log = hstack([X_text_train, csr_matrix(X_num_logstd_train), csr_matrix(X_bin_train)])
X_test_all_log = hstack([X_text_test, csr_matrix(X_num_logstd_test), csr_matrix(X_bin_test)])

print(f"\nLog+StandardScaler -> Combined matrix: {X_all_log.shape}")

# %%
# --------------------------------------------------
# STEP 4: Train-Validation Split (on precomputed matrices)
# --------------------------------------------------
from sklearn.model_selection import train_test_split

indices = np.arange(len(y))
idx_train, idx_val = train_test_split(
    indices, test_size=0.2, stratify=y, random_state=42
)

# Standard Scaler split
X_tr_std, X_vl_std = X_all_std[idx_train], X_all_std[idx_val]

# MinMax split
X_tr_mm, X_vl_mm = X_all_mm[idx_train], X_all_mm[idx_val]

# Log+Standard split
X_tr_log, X_vl_log = X_all_log[idx_train], X_all_log[idx_val]

y_tr, y_vl = y[idx_train], y[idx_val]

print(f"Split: Train={len(y_tr):,} | Val={len(y_vl):,}")
print(f"Label distribution (train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

# %% [markdown]
# ---
# ## 6. Evaluation Helper & Results Tracker

# %%
all_results = []

def evaluate_model(name, y_true, y_pred, detailed=True):
    """Evaluate and store results."""
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_wt  = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro F1:       {f1_mac:.4f}")
    print(f"  Weighted F1:    {f1_wt:.4f}")

    if detailed:
        print(f"\n{classification_report(y_true, y_pred, digits=4)}")

    result = {"name": name, "accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt}
    all_results.append(result)
    return result

# %% [markdown]
# ---
# ## 7. Scaling Comparison — Logistic Regression

# %%
# LR with StandardScaler
print("LR (StandardScaler)...")
t0 = time.time()
lr_std = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", C=2.0, n_jobs=-1, random_state=42)
lr_std.fit(X_tr_std, y_tr)
evaluate_model("LR (StandardScaler)", y_vl, lr_std.predict(X_vl_std))
print(f"  ⏱️ Time: {time.time()-t0:.1f}s")

# %%
# LR with MinMaxScaler
print("LR (MinMaxScaler)...")
t0 = time.time()
lr_mm = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", C=2.0, n_jobs=-1, random_state=42)
lr_mm.fit(X_tr_mm, y_tr)
evaluate_model("LR (MinMaxScaler)", y_vl, lr_mm.predict(X_vl_mm))
print(f"  ⏱️ Time: {time.time()-t0:.1f}s")

# %%
# LR with Log+StandardScaler
print("LR (Log+StandardScaler)...")
t0 = time.time()
lr_log = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", C=2.0, n_jobs=-1, random_state=42)
lr_log.fit(X_tr_log, y_tr)
evaluate_model("LR (Log+StandardScaler)", y_vl, lr_log.predict(X_vl_log))
print(f"  ⏱️ Time: {time.time()-t0:.1f}s")

# %%
# Scaling comparison summary
print("\n" + "=" * 70)
print("  SCALING COMPARISON SUMMARY")
print("=" * 70)
scaling_df = pd.DataFrame(all_results[-3:])
print(scaling_df[["name", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))
best_scaler_name = scaling_df.loc[scaling_df["macro_f1"].idxmax(), "name"]
print(f"\n-> Best scaler: {best_scaler_name}")

# %% [markdown]
# ---
# ## 8. Model 1: Tuned Logistic Regression (GridSearchCV)
#
# Using the best performing scaler. GridSearchCV is now **much faster** because
# TF-IDF is precomputed — only the classifier is being refit in each fold.

# %%
# Determine best feature matrix
# (Using StandardScaler as default; will switch if another wins)
scaler_map = {
    "LR (StandardScaler)": (X_all_std, X_test_all_std),
    "LR (MinMaxScaler)": (X_all_mm, X_test_all_mm),
    "LR (Log+StandardScaler)": (X_all_log, X_test_all_log),
}

X_best, X_test_best = scaler_map.get(best_scaler_name, (X_all_std, X_test_all_std))
print(f"Using feature matrix from: {best_scaler_name}")
print(f"Matrix shape: {X_best.shape}")

# %%
print("GridSearchCV for Logistic Regression...")
t0 = time.time()

f1_macro_scorer = make_scorer(f1_score, average="macro")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid_lr = {
    "C": [1.0, 2.0, 5.0],
    "solver": ["lbfgs"],       # lbfgs is much faster than saga for L2 on sparse data
    "penalty": ["l2"],
    "max_iter": [3000]
}

grid_lr = GridSearchCV(
    LogisticRegression(class_weight="balanced", n_jobs=-1, random_state=42),
    param_grid_lr,
    cv=cv,
    scoring=f1_macro_scorer,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_lr.fit(X_best, y)
print(f"\nGridSearchCV done in {time.time()-t0:.1f}s")
print(f"Best params: {grid_lr.best_params_}")
print(f"Best CV Macro F1: {grid_lr.best_score_:.4f}")

# %%
# Evaluate on validation split
evaluate_model("Tuned LR (GridSearchCV)", y_vl, grid_lr.predict(X_best[idx_val]))

# %%
# Grid results table
gs_results = pd.DataFrame(grid_lr.cv_results_)
print("\nAll GridSearch Results:")
print(gs_results[["param_C", "param_solver",
                   "mean_test_score", "std_test_score", "rank_test_score"]]
      .sort_values("rank_test_score").to_string(index=False))

# %% [markdown]
# ---
# ## 9. Model 2: SGDClassifier (log_loss)
#
# SGD with `log_loss` = online-learning Logistic Regression. Much faster for large data.

# %%
print("SGDClassifier (log_loss) — default...")
t0 = time.time()

sgd_log = SGDClassifier(
    loss="log_loss", class_weight="balanced",
    max_iter=2000, tol=1e-4, random_state=42, n_jobs=-1
)
sgd_log.fit(X_tr_std, y_tr)
evaluate_model("SGD (log_loss) default", y_vl, sgd_log.predict(X_vl_std))
print(f"  ⏱️ Time: {time.time()-t0:.1f}s")

# %%
# Hyperparameter Tuning
print("RandomizedSearchCV for SGD (log_loss)...")
t0 = time.time()

param_dist_sgd_log = {
    "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "penalty": ["l2", "l1", "elasticnet"],
    "max_iter": [2000, 3000],
    "learning_rate": ["optimal", "adaptive"],
    "eta0": [0.01, 0.1],
}

rs_sgd_log = RandomizedSearchCV(
    SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42, n_jobs=-1),
    param_dist_sgd_log,
    n_iter=12,
    cv=cv,
    scoring=f1_macro_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True
)

rs_sgd_log.fit(X_best, y)
print(f"\nDone in {time.time()-t0:.1f}s")
print(f"Best params: {rs_sgd_log.best_params_}")
print(f"Best CV Macro F1: {rs_sgd_log.best_score_:.4f}")

# %%
evaluate_model("Tuned SGD (log_loss)", y_vl, rs_sgd_log.predict(X_best[idx_val]))

# %% [markdown]
# ---
# ## 10. Model 3: SGDClassifier (hinge loss — Linear SVM)
#
# `hinge` loss trains a linear Support Vector Machine using SGD.
# Much faster than `sklearn.svm.SVC` for 198K rows.

# %%
print("SGDClassifier (hinge) — default...")
t0 = time.time()

sgd_hinge = SGDClassifier(
    loss="hinge", class_weight="balanced",
    max_iter=2000, tol=1e-4, random_state=42, n_jobs=-1
)
sgd_hinge.fit(X_tr_std, y_tr)
evaluate_model("SGD (hinge/SVM) default", y_vl, sgd_hinge.predict(X_vl_std))
print(f"  ⏱️ Time: {time.time()-t0:.1f}s")

# %%
# Hyperparameter Tuning
print("RandomizedSearchCV for SGD (hinge)...")
t0 = time.time()

param_dist_sgd_hinge = {
    "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "penalty": ["l2", "l1", "elasticnet"],
    "max_iter": [2000, 3000],
    "learning_rate": ["optimal", "adaptive"],
    "eta0": [0.01, 0.1],
}

rs_sgd_hinge = RandomizedSearchCV(
    SGDClassifier(loss="hinge", class_weight="balanced", random_state=42, n_jobs=-1),
    param_dist_sgd_hinge,
    n_iter=12,
    cv=cv,
    scoring=f1_macro_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True
)

rs_sgd_hinge.fit(X_best, y)
print(f"\nDone in {time.time()-t0:.1f}s")
print(f"Best params: {rs_sgd_hinge.best_params_}")
print(f"Best CV Macro F1: {rs_sgd_hinge.best_score_:.4f}")

# %%
evaluate_model("Tuned SGD (hinge/SVM)", y_vl, rs_sgd_hinge.predict(X_best[idx_val]))

# %% [markdown]
# ---
# ## 11. Model 4: RidgeClassifier
#
# Fast L2-regularized linear model. Treats classification as regression,
# then uses the sign for prediction. Often surprisingly competitive.

# %%
print("RidgeClassifier — default...")
t0 = time.time()

ridge = RidgeClassifier(class_weight="balanced", alpha=1.0)
ridge.fit(X_tr_std, y_tr)
evaluate_model("RidgeClassifier default", y_vl, ridge.predict(X_vl_std))
print(f"  ⏱️ Time: {time.time()-t0:.1f}s")

# %%
# Hyperparameter Tuning
print("GridSearchCV for RidgeClassifier...")
t0 = time.time()

param_grid_ridge = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
}

grid_ridge = GridSearchCV(
    RidgeClassifier(class_weight="balanced"),
    param_grid_ridge,
    cv=cv,
    scoring=f1_macro_scorer,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_ridge.fit(X_best, y)
print(f"\nDone in {time.time()-t0:.1f}s")
print(f"Best alpha: {grid_ridge.best_params_}")
print(f"Best CV Macro F1: {grid_ridge.best_score_:.4f}")

# %%
evaluate_model("Tuned RidgeClassifier", y_vl, grid_ridge.predict(X_best[idx_val]))

# %%
# Ridge alpha curve
ridge_res = pd.DataFrame(grid_ridge.cv_results_)
plt.figure(figsize=(10, 5))
plt.plot(ridge_res["param_alpha"].astype(float),
         ridge_res["mean_test_score"], marker="o", linewidth=2, color="steelblue")
plt.fill_between(
    ridge_res["param_alpha"].astype(float),
    ridge_res["mean_test_score"] - ridge_res["std_test_score"],
    ridge_res["mean_test_score"] + ridge_res["std_test_score"],
    alpha=0.2, color="steelblue"
)
plt.xlabel("Alpha (Regularization)")
plt.ylabel("Mean CV Macro F1")
plt.title("RidgeClassifier: Alpha Tuning Curve", fontsize=14, fontweight="bold")
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 12. Complete Model Comparison

# %%
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("  COMPLETE MODEL COMPARISON — Milestone 2")
print("=" * 80)
print(results_df.to_string(index=False))

# %%
# Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

metrics = ["accuracy", "macro_f1", "weighted_f1"]
titles  = ["Accuracy", "Macro F1-Score", "Weighted F1-Score"]
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    bars = axes[idx].barh(results_df["name"], results_df[metric],
                          color=colors_bar, edgecolor="black")
    axes[idx].set_title(title, fontsize=13, fontweight="bold")
    axes[idx].set_xlim(results_df[metric].min() - 0.03, results_df[metric].max() + 0.02)
    for bar, val in zip(bars, results_df[metric]):
        axes[idx].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                      f"{val:.4f}", va="center", fontsize=8)

plt.suptitle("Milestone 2: Model Comparison", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Confusion Matrices — All Tuned Models

# %%
tuned_preds = {
    "Tuned LR": grid_lr.predict(X_best[idx_val]),
    "Tuned SGD (log)": rs_sgd_log.predict(X_best[idx_val]),
    "Tuned SGD (hinge)": rs_sgd_hinge.predict(X_best[idx_val]),
    "Tuned Ridge": grid_ridge.predict(X_best[idx_val]),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (name, preds) in enumerate(tuned_preds.items()):
    ax = axes[idx // 2][idx % 2]
    cm = confusion_matrix(y_vl, preds, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3]).plot(
        ax=ax, cmap="Blues", values_format=".2f"
    )
    ax.set_title(name, fontsize=12, fontweight="bold")

plt.suptitle("Normalized Confusion Matrices — Tuned Models",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 14. Generate Submission with Best Model

# %%
# Identify best
best_idx = results_df["macro_f1"].idxmax()
best_name = results_df.loc[best_idx, "name"]
best_f1   = results_df.loc[best_idx, "macro_f1"]
print(f"Best Model: {best_name} (Macro F1: {best_f1:.4f})")

# %%
# Build mapping from name to fitted model
tuned_models = {
    "Tuned LR (GridSearchCV)": grid_lr.best_estimator_,
    "Tuned SGD (log_loss)": rs_sgd_log.best_estimator_,
    "Tuned SGD (hinge/SVM)": rs_sgd_hinge.best_estimator_,
    "Tuned RidgeClassifier": grid_ridge.best_estimator_,
}

# GridSearchCV/RandomizedSearchCV already retrained on full data (refit=True)
# so we can predict directly

# %%
# Find best fitted model
# Try to match the best name to a tuned model
best_model = None
for key, model in tuned_models.items():
    if key in best_name or best_name in key:
        best_model = model
        print(f"Using tuned model: {key}")
        break

if best_model is None:
    # Fallback: retrain best grid search winner on full data
    print("Falling back to GridSearchCV LR best estimator")
    best_model = grid_lr.best_estimator_
    best_model.fit(X_best, y)

# %%
# Predict on test
test_predictions = best_model.predict(X_test_best)

submission = sample.copy()
submission["label"] = test_predictions

print("Submission preview:")
print(submission.head(10))
print(f"\nPrediction distribution:")
print(pd.Series(test_predictions).value_counts().sort_index())

# %%
submission.to_csv("submission.csv", index=False)
print("Submission saved to 'submission.csv'")

# %% [markdown]
# ---
# ## 15. Milestone 2 Summary
#
# ### Optimization Applied:
# TF-IDF precomputed ONCE -> reused across all models and CV folds.
# **Estimated runtime: ~15-25 minutes** (vs ~2 hours with pipelines).
#
# ### Models Trained & Tuned:
# | # | Model | Tuning | Status |
# |---|-------|--------|--------|
# | 1 | Logistic Regression | GridSearchCV (C, solver) | |
# | 2 | SGDClassifier (log_loss) | RandomizedSearchCV (alpha, penalty, lr) | |
# | 3 | SGDClassifier (hinge = Linear SVM) | RandomizedSearchCV | |
# | 4 | RidgeClassifier | GridSearchCV (alpha) | |
#
# ### Scaling Experiments:
# StandardScaler vs MinMaxScaler vs Log1p+StandardScaler
#
# ### M2 Question Answers:
# All 10 answers printed inline in Section 3 with `[Q] M2 Q` markers.
#
# ---
# **Next:** Milestone 3 — Naive Bayes, KNN, SVM, Dimensionality Reduction
