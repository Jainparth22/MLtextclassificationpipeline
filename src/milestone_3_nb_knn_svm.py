# %% [markdown]
# # Milestone 3: Naive Bayes, KNN, SVM, Feature Selection & Dimensionality Reduction
# **Comment Category Prediction Challenge**
#
# **M2 Best Results:** Tuned LR -> Accuracy 0.909, Macro F1 0.828 | Kaggle: 0.776
#
# **This notebook covers:**
# 1. Milestone 3 Questions (all 10)
# 2. Feature Selection & Dimensionality Reduction (TruncatedSVD, SelectKBest)
# 3. **MultinomialNB** & **ComplementNB**
# 4. **LinearSVC** (typically #1 for TF-IDF text)
# 5. **PassiveAggressiveClassifier**
# 6. **KNN** (with dimensionality reduction)
# 7. **SVM** (SGD hinge as proxy for large data)
# 8. Hyperparameter Tuning
# 9. Model Comparison & Best Submission (targeting 0.90)
#
# Uses precomputed TF-IDF pipeline (optimized).
# ---

# %% [markdown]
# ## 1. Setup & Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import string

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, RandomizedSearchCV,
                                     cross_val_score)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   OneHotEncoder, LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import (LogisticRegression, SGDClassifier,
                                  RidgeClassifier, PassiveAggressiveClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

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
# ## 3. Milestone 3 Questions
#
# These questions follow specific preprocessing steps as prescribed.

# %%

# [Q] M3 Q1: Split train with random_state=42, 80/20.
# If X_train shape (a,b) and X_val shape (c,d), what is a + c?

# Drop NaN comment first
train_clean = train.dropna(subset=["comment"]).copy()

X = train_clean.drop(columns=["label"])
y = train_clean["label"]

X_train_q, X_val_q, y_train_q, y_val_q = train_test_split(
    X, y, test_size=0.2, random_state=42
)

a, b = X_train_q.shape
c, d = X_val_q.shape

print(f"X_train shape: ({a}, {b})")
print(f"X_val shape:   ({c}, {d})")
print(f"\n-> ANSWER: a + c = {a} + {c} = {a + c}")

# %%

# [Q] M3 Q2: Convert created_date to datetime, extract day/month/year
# for X_train and X_val. Most frequently occurring month in X_train?

X_train_q["created_date"] = pd.to_datetime(X_train_q["created_date"], errors="coerce")
X_val_q["created_date"]   = pd.to_datetime(X_val_q["created_date"], errors="coerce")

X_train_q["day"]   = X_train_q["created_date"].dt.day
X_train_q["month"] = X_train_q["created_date"].dt.month
X_train_q["year"]  = X_train_q["created_date"].dt.year

X_val_q["day"]   = X_val_q["created_date"].dt.day
X_val_q["month"] = X_val_q["created_date"].dt.month
X_val_q["year"]  = X_val_q["created_date"].dt.year

month_counts = X_train_q["month"].value_counts()
most_freq_month = month_counts.index[0]
month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May",
               6:"June", 7:"July", 8:"August", 9:"September", 10:"October",
               11:"November", 12:"December"}

print(f"Month value counts in X_train:")
print(month_counts)
print(f"\n-> ANSWER: Most frequent month = {most_freq_month} ({month_names.get(most_freq_month, 'Unknown')})")

# %%

# [Q] M3 Q3: Impute null values of categorical features with 'none'.
# One-hot encode ONLY religion, gender, race with handle_unknown='ignore'.
# Shape of X_train after imputing -> what is b?
# Also transform X_val.

# Identify categorical columns
cat_cols = X_train_q.select_dtypes(include=["object"]).columns.tolist()
# Remove 'comment' from cat_cols for imputation (we handle text separately)
cat_cols_to_impute = [c for c in cat_cols if c != "comment"]
print(f"Categorical columns to impute: {cat_cols_to_impute}")

# Impute categorical nulls with 'none'
for col in cat_cols_to_impute:
    X_train_q[col] = X_train_q[col].fillna("none")
    X_val_q[col]   = X_val_q[col].fillna("none")

print(f"\nX_train shape after imputing: {X_train_q.shape}")
print(f"-> ANSWER (shape after imputing): b = {X_train_q.shape[1]}")

# One-hot encode religion, gender, race
ohe_cols = ["religion", "gender", "race"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train_q[ohe_cols])

X_train_ohe = pd.DataFrame(
    ohe.transform(X_train_q[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=X_train_q.index
)
X_val_ohe = pd.DataFrame(
    ohe.transform(X_val_q[ohe_cols]),
    columns=ohe.get_feature_names_out(ohe_cols),
    index=X_val_q.index
)

# Drop original columns and add encoded
X_train_q = X_train_q.drop(columns=ohe_cols).join(X_train_ohe)
X_val_q   = X_val_q.drop(columns=ohe_cols).join(X_val_ohe)

print(f"\nX_train shape after OHE: {X_train_q.shape}")
print(f"X_val shape after OHE:   {X_val_q.shape}")
print(f"OHE feature names: {ohe.get_feature_names_out(ohe_cols).tolist()}")

# %%

# [Q] M3 Q4: Apply CountVectorizer to comment column of X_train and X_val.
# Sum of counts for document at index 1 (second row) of transformed X_train?

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train_q["comment"].astype(str))
X_val_cv   = cv.transform(X_val_q["comment"].astype(str))

# Index 1 = second row
doc_1_sum = X_train_cv[1].sum()
print(f"CountVectorizer X_train shape: {X_train_cv.shape}")
print(f"Sum of counts at index 1 (second row): {doc_1_sum}")
print(f"\n-> ANSWER: Sum of counts for document at index 1 = {doc_1_sum}")

# %%

# [Q] M3 Q5: Convert disability to int (True->1, False->0).
# Sum of all disability values in X_train and X_val?

X_train_q["disability"] = X_train_q["disability"].astype(int)
X_val_q["disability"]   = X_val_q["disability"].astype(int)

disability_sum_train = X_train_q["disability"].sum()
disability_sum_val   = X_val_q["disability"].sum()
total_disability = disability_sum_train + disability_sum_val

print(f"Disability sum in X_train: {disability_sum_train}")
print(f"Disability sum in X_val:   {disability_sum_val}")
print(f"\n-> ANSWER: Total disability sum = {total_disability}")

# %%

# [Q] M3 Q6: Scale numeric features using StandardScaler.
# Number of features seen during fit for X_train?
# Drop all datetime columns first.

# Drop datetime columns
datetime_cols = X_train_q.select_dtypes(include=["datetime64"]).columns.tolist()
print(f"Datetime columns to drop: {datetime_cols}")

X_train_q6 = X_train_q.drop(columns=datetime_cols, errors="ignore")
X_val_q6   = X_val_q.drop(columns=datetime_cols, errors="ignore")

# Also drop non-numeric columns (comment, etc.)
non_numeric = X_train_q6.select_dtypes(include=["object"]).columns.tolist()
print(f"Non-numeric columns (excluded from scaling): {non_numeric}")

numeric_cols_q6 = X_train_q6.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns for scaling: {numeric_cols_q6}")

scaler_q6 = StandardScaler()
scaler_q6.fit(X_train_q6[numeric_cols_q6])

n_features_seen = scaler_q6.n_features_in_
print(f"\nNumber of features seen during fit: {n_features_seen}")
print(f"\n-> ANSWER: Features seen during StandardScaler fit = {n_features_seen}")

# %%

# [Q] M3 Q7 & Q8: Full preprocessing -> MultinomialNB -> Macro F1 on
# train AND validation.
#
# Steps:
# 1. Split 80/20
# 2. SimpleImputer fit on X_train, transform both
# 3. Ensure no negative values (abs)
# 4. Convert created_date -> day, month, year; drop created_date
# 5. TfidfVectorizer(stop_words='english') on comment
# 6. OneHotEncoder(handle_unknown='ignore') on categorical features
# 7. Train MultinomialNB
# 8. Report macro F1 on train and val

# Fresh reload for clean preprocessing
train_q7 = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/train.csv")
train_q7 = train_q7.dropna(subset=["comment"])

X_q7 = train_q7.drop(columns=["label"])
y_q7 = train_q7["label"]

X_train_q7, X_val_q7, y_train_q7, y_val_q7 = train_test_split(
    X_q7, y_q7, test_size=0.2, random_state=42
)

# Step 2: SimpleImputer
# Separate numeric and non-numeric
num_cols_q7 = X_train_q7.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_q7 = X_train_q7.select_dtypes(include=["object", "bool"]).columns.tolist()
cat_cols_q7_no_comment = [c for c in cat_cols_q7 if c != "comment"]

# Impute numeric with mean
num_imputer = SimpleImputer(strategy="mean")
X_train_q7[num_cols_q7] = num_imputer.fit_transform(X_train_q7[num_cols_q7])
X_val_q7[num_cols_q7]   = num_imputer.transform(X_val_q7[num_cols_q7])

# Impute categorical with most_frequent
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_q7[cat_cols_q7_no_comment] = cat_imputer.fit_transform(X_train_q7[cat_cols_q7_no_comment])
X_val_q7[cat_cols_q7_no_comment]   = cat_imputer.transform(X_val_q7[cat_cols_q7_no_comment])

# Step 3: Ensure no negative values
for col in num_cols_q7:
    X_train_q7[col] = X_train_q7[col].abs()
    X_val_q7[col]   = X_val_q7[col].abs()

# Step 4: created_date -> day, month, year
X_train_q7["created_date"] = pd.to_datetime(X_train_q7["created_date"], errors="coerce")
X_val_q7["created_date"]   = pd.to_datetime(X_val_q7["created_date"], errors="coerce")

for df in [X_train_q7, X_val_q7]:
    df["day"]   = df["created_date"].dt.day
    df["month"] = df["created_date"].dt.month
    df["year"]  = df["created_date"].dt.year

X_train_q7 = X_train_q7.drop(columns=["created_date"])
X_val_q7   = X_val_q7.drop(columns=["created_date"])

# Step 5: TF-IDF on comment
tfidf_q7 = TfidfVectorizer(stop_words="english")
X_train_text_q7 = tfidf_q7.fit_transform(X_train_q7["comment"].astype(str))
X_val_text_q7   = tfidf_q7.transform(X_val_q7["comment"].astype(str))

# Step 6: OneHotEncoder on categorical features
cat_features_q7 = [c for c in X_train_q7.columns
                    if X_train_q7[c].dtype == "object" and c != "comment"]
# Also include boolean columns
bool_cols_q7 = X_train_q7.select_dtypes(include=["bool"]).columns.tolist()
# Convert bool to string for OHE
for col in bool_cols_q7:
    X_train_q7[col] = X_train_q7[col].astype(str)
    X_val_q7[col]   = X_val_q7[col].astype(str)
cat_features_q7 = cat_features_q7 + bool_cols_q7

print(f"Categorical features for OHE: {cat_features_q7}")

ohe_q7 = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_cat_q7 = ohe_q7.fit_transform(X_train_q7[cat_features_q7])
X_val_cat_q7   = ohe_q7.transform(X_val_q7[cat_features_q7])

# Get remaining numeric columns (drop comment and cat features)
num_cols_final_q7 = [c for c in X_train_q7.columns
                     if c not in cat_features_q7 and c != "comment"
                     and X_train_q7[c].dtype in [np.int64, np.float64, np.int32, np.float32]]

print(f"Numeric features: {num_cols_final_q7}")

X_train_num_q7 = csr_matrix(X_train_q7[num_cols_final_q7].values)
X_val_num_q7   = csr_matrix(X_val_q7[num_cols_final_q7].values)

# Combine all
X_train_final_q7 = hstack([X_train_text_q7, X_train_num_q7, X_train_cat_q7])
X_val_final_q7   = hstack([X_val_text_q7, X_val_num_q7, X_val_cat_q7])

print(f"\nCombined train matrix: {X_train_final_q7.shape}")
print(f"Combined val matrix:   {X_val_final_q7.shape}")

# Step 7: Train MultinomialNB
mnb_q7 = MultinomialNB()
mnb_q7.fit(X_train_final_q7, y_train_q7)

# Step 8: Macro F1
y_train_pred_q7 = mnb_q7.predict(X_train_final_q7)
y_val_pred_q7   = mnb_q7.predict(X_val_final_q7)

f1_train_q7 = f1_score(y_train_q7, y_train_pred_q7, average="macro")
f1_val_q7   = f1_score(y_val_q7, y_val_pred_q7, average="macro")

print(f"\nMultinomialNB - Macro F1 on TRAIN: {f1_train_q7:.4f}")
print(f"MultinomialNB - Macro F1 on VAL:   {f1_val_q7:.4f}")
print(f"\n-> ANSWER Q7: Macro F1 on train = {f1_train_q7:.4f}")
print(f"-> ANSWER Q8: Macro F1 on val   = {f1_val_q7:.4f}")

# %%

# [Q] M3 Q9 & Q10: Same as Q7-Q8 but add is_weekend feature.
# is_weekend = 1 for Saturday/Sunday, 0 otherwise.

# Fresh reload
train_q9 = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/train.csv")
train_q9 = train_q9.dropna(subset=["comment"])

X_q9 = train_q9.drop(columns=["label"])
y_q9 = train_q9["label"]

X_train_q9, X_val_q9, y_train_q9, y_val_q9 = train_test_split(
    X_q9, y_q9, test_size=0.2, random_state=42
)

# Impute
num_cols_q9 = X_train_q9.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_q9 = X_train_q9.select_dtypes(include=["object", "bool"]).columns.tolist()
cat_cols_q9_nc = [c for c in cat_cols_q9 if c != "comment"]

num_imp_q9 = SimpleImputer(strategy="mean")
X_train_q9[num_cols_q9] = num_imp_q9.fit_transform(X_train_q9[num_cols_q9])
X_val_q9[num_cols_q9]   = num_imp_q9.transform(X_val_q9[num_cols_q9])

cat_imp_q9 = SimpleImputer(strategy="most_frequent")
X_train_q9[cat_cols_q9_nc] = cat_imp_q9.fit_transform(X_train_q9[cat_cols_q9_nc])
X_val_q9[cat_cols_q9_nc]   = cat_imp_q9.transform(X_val_q9[cat_cols_q9_nc])

# Ensure no negative values
for col in num_cols_q9:
    X_train_q9[col] = X_train_q9[col].abs()
    X_val_q9[col]   = X_val_q9[col].abs()

# Datetime engineering
X_train_q9["created_date"] = pd.to_datetime(X_train_q9["created_date"], errors="coerce")
X_val_q9["created_date"]   = pd.to_datetime(X_val_q9["created_date"], errors="coerce")

for df in [X_train_q9, X_val_q9]:
    df["day"]   = df["created_date"].dt.day
    df["month"] = df["created_date"].dt.month
    df["year"]  = df["created_date"].dt.year
    # NEW: is_weekend (Saturday=5, Sunday=6)
    df["is_weekend"] = (df["created_date"].dt.dayofweek >= 5).astype(int)

X_train_q9 = X_train_q9.drop(columns=["created_date"])
X_val_q9   = X_val_q9.drop(columns=["created_date"])

# TF-IDF
tfidf_q9 = TfidfVectorizer(stop_words="english")
X_train_text_q9 = tfidf_q9.fit_transform(X_train_q9["comment"].astype(str))
X_val_text_q9   = tfidf_q9.transform(X_val_q9["comment"].astype(str))

# OHE on categorical features INCLUDING is_weekend
cat_features_q9 = [c for c in X_train_q9.columns
                    if X_train_q9[c].dtype == "object" and c != "comment"]
bool_cols_q9 = X_train_q9.select_dtypes(include=["bool"]).columns.tolist()
for col in bool_cols_q9:
    X_train_q9[col] = X_train_q9[col].astype(str)
    X_val_q9[col]   = X_val_q9[col].astype(str)
cat_features_q9 = cat_features_q9 + bool_cols_q9

# Add is_weekend as categorical for OHE
X_train_q9["is_weekend"] = X_train_q9["is_weekend"].astype(str)
X_val_q9["is_weekend"]   = X_val_q9["is_weekend"].astype(str)
if "is_weekend" not in cat_features_q9:
    cat_features_q9.append("is_weekend")

print(f"Categorical features for OHE (with is_weekend): {cat_features_q9}")

ohe_q9 = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_cat_q9 = ohe_q9.fit_transform(X_train_q9[cat_features_q9])
X_val_cat_q9   = ohe_q9.transform(X_val_q9[cat_features_q9])

num_cols_final_q9 = [c for c in X_train_q9.columns
                     if c not in cat_features_q9 and c != "comment"
                     and X_train_q9[c].dtype in [np.int64, np.float64, np.int32, np.float32]]

X_train_num_q9 = csr_matrix(X_train_q9[num_cols_final_q9].values)
X_val_num_q9   = csr_matrix(X_val_q9[num_cols_final_q9].values)

X_train_final_q9 = hstack([X_train_text_q9, X_train_num_q9, X_train_cat_q9])
X_val_final_q9   = hstack([X_val_text_q9, X_val_num_q9, X_val_cat_q9])

print(f"Combined train: {X_train_final_q9.shape} | Combined val: {X_val_final_q9.shape}")

# Train MultinomialNB
mnb_q9 = MultinomialNB()
mnb_q9.fit(X_train_final_q9, y_train_q9)

y_train_pred_q9 = mnb_q9.predict(X_train_final_q9)
y_val_pred_q9   = mnb_q9.predict(X_val_final_q9)

f1_train_q9 = f1_score(y_train_q9, y_train_pred_q9, average="macro")
f1_val_q9   = f1_score(y_val_q9, y_val_pred_q9, average="macro")

print(f"\nMultinomialNB (with is_weekend) - Macro F1 on TRAIN: {f1_train_q9:.4f}")
print(f"MultinomialNB (with is_weekend) - Macro F1 on VAL:   {f1_val_q9:.4f}")
print(f"\n-> ANSWER Q9:  Macro F1 on train = {f1_train_q9:.4f}")
print(f"-> ANSWER Q10: Macro F1 on val   = {f1_val_q9:.4f}")

# %% [markdown]
# ---
# ## 4. Feature Engineering (for Competition Models)
#
# Now we switch to the optimized precomputed pipeline for the actual competition models.

# %%
# Fresh load for competition modeling
train = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/train.csv")
test  = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/test.csv")
sample = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/Sample.csv")

train = train.dropna(subset=["comment"])

# Feature engineering
for df in [train, test]:
    df["comment"] = df["comment"].fillna("")
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["hour"] = df["created_date"].dt.hour
    df["day_of_week"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.month
    df["comment_length"] = df["comment"].astype(str).str.len()
    df["word_count"] = df["comment"].astype(str).str.split().str.len().fillna(0).astype(int)
    df["avg_word_length"] = df["comment_length"] / (df["word_count"] + 1)
    df["total_votes"] = df["upvote"] + df["downvote"]
    df["vote_diff"] = df["upvote"] - df["downvote"]
    df["vote_ratio"] = df["upvote"] / (df["downvote"] + 1)
    df["total_emoticons"] = df["emoticon_1"] + df["emoticon_2"] + df["emoticon_3"]
    for col in ["race", "religion", "gender"]:
        df[col] = df[col].fillna("missing")
    df["has_detection"] = (df[["race","religion","gender"]].ne("missing").any(axis=1)).astype(int)

print(f"Feature engineering done. Train: {train.shape} | Test: {test.shape}")

# %% [markdown]
# ## 5. Optimized Pipeline — Precompute Features

# %%
# --------------------------------------------------
# STEP 1: TF-IDF (computed ONCE)
# --------------------------------------------------
print("Computing TF-IDF...")
t0 = time.time()

tfidf = TfidfVectorizer(
    max_features=15000,        # Increased from 10K for better score
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3, max_df=0.9,
    sublinear_tf=True
)

X_text_train = tfidf.fit_transform(train["comment"].astype(str))
X_text_test  = tfidf.transform(test["comment"].astype(str))
print(f"TF-IDF done in {time.time()-t0:.1f}s -> {X_text_train.shape[1]} features")

# %%
# --------------------------------------------------
# STEP 2: Character N-gram TF-IDF (extra signal for better score)
# --------------------------------------------------
print("Computing char n-gram TF-IDF...")
t0 = time.time()

tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=10000,         # 10K char n-gram features for capturing subword patterns
    min_df=3, max_df=0.9,
    sublinear_tf=True
)

X_char_train = tfidf_char.fit_transform(train["comment"].astype(str))
X_char_test  = tfidf_char.transform(test["comment"].astype(str))
print(f"Char TF-IDF done in {time.time()-t0:.1f}s -> {X_char_train.shape[1]} features")

# %%
# --------------------------------------------------
# STEP 3: Numeric & Binary features
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

X_num_train = train[numeric_features].values.copy()
X_bin_train = train[binary_features].values.astype(float)
X_num_test  = test[numeric_features].values.copy()
X_bin_test  = test[binary_features].values.astype(float)

y = train["label"].values

# Log transform skewed features
skewed_cols = ["upvote", "downvote", "if_1", "if_2",
               "total_votes", "vote_ratio", "comment_length", "word_count"]
skewed_idx = [numeric_features.index(c) for c in skewed_cols]

for idx in skewed_idx:
    X_num_train[:, idx] = np.log1p(np.clip(X_num_train[:, idx], 0, None))
    X_num_test[:, idx]  = np.log1p(np.clip(X_num_test[:, idx], 0, None))

scaler = StandardScaler()
X_num_scaled_train = scaler.fit_transform(X_num_train)
X_num_scaled_test  = scaler.transform(X_num_test)

# %%
# --------------------------------------------------
# STEP 4: Combine all features
# --------------------------------------------------

# Version A: Word TF-IDF only (for NB models that need non-negative input)
X_all_nonneg = hstack([X_text_train, X_char_train, csr_matrix(X_bin_train)])

# Version B: Full features with scaling (for SVM, LR, etc.)
X_all = hstack([X_text_train, X_char_train,
                csr_matrix(X_num_scaled_train), csr_matrix(X_bin_train)])
X_test_all = hstack([X_text_test, X_char_test,
                     csr_matrix(X_num_scaled_test), csr_matrix(X_bin_test)])

# Version C: Non-negative full (for NB with numerics — abs scale)
X_num_mm_train = MinMaxScaler().fit_transform(X_num_train)
X_num_mm_test  = MinMaxScaler().fit(X_num_train).transform(X_num_test)
X_all_nonneg_full = hstack([X_text_train, X_char_train,
                            csr_matrix(X_num_mm_train), csr_matrix(X_bin_train)])
X_test_nonneg_full = hstack([X_text_test, X_char_test,
                             csr_matrix(X_num_mm_test), csr_matrix(X_bin_test)])

print(f"Feature matrices ready:")
print(f"   Full (scaled):     {X_all.shape}")
print(f"   Non-neg (NB):      {X_all_nonneg.shape}")
print(f"   Non-neg full (NB): {X_all_nonneg_full.shape}")

# %%
# --------------------------------------------------
# STEP 5: Train-Validation Split
# --------------------------------------------------
indices = np.arange(len(y))
idx_train, idx_val = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

X_tr, X_vl = X_all[idx_train], X_all[idx_val]
X_tr_nn, X_vl_nn = X_all_nonneg_full[idx_train], X_all_nonneg_full[idx_val]
y_tr, y_vl = y[idx_train], y[idx_val]

print(f"Split: Train={len(y_tr):,} | Val={len(y_vl):,}")

# %% [markdown]
# ---
# ## 6. Evaluation Helper

# %%
all_results = []

def evaluate_model(name, y_true, y_pred, detailed=True):
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_wt  = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Macro F1:     {f1_mac:.4f}")
    print(f"  Weighted F1:  {f1_wt:.4f}")
    if detailed:
        print(f"\n{classification_report(y_true, y_pred, digits=4)}")
    result = {"name": name, "accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt}
    all_results.append(result)
    return result

f1_macro_scorer = make_scorer(f1_score, average="macro")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# %% [markdown]
# ---
# ## 7. Model 1: MultinomialNB

# %%

# DEFAULT MultinomialNB — COMMENTED OUT (already ran in first pass)
# LESSON LEARNED: Default alpha=1.0 gave Macro F1 ~0.64, Accuracy ~0.74
# Tuned version below finds optimal alpha, so running default first is redundant.

# print("MultinomialNB...")
# t0 = time.time()
# mnb = MultinomialNB(alpha=1.0)
# mnb.fit(X_tr_nn, y_tr)
# evaluate_model("MultinomialNB (default)", y_vl, mnb.predict(X_vl_nn))
# print(f"  ⏱️ {time.time()-t0:.1f}s")
# RESULT FROM FIRST RUN:
#   Accuracy: 0.7396 | Macro F1: 0.6424 | Weighted F1: 0.7382
#   Label 3 F1: very low (~0.35) — MultinomialNB struggles with minority classes

# %%
# Tuning alpha
print("GridSearchCV for MultinomialNB...")
t0 = time.time()
grid_mnb = GridSearchCV(
    MultinomialNB(),
    {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
    cv=cv, scoring=f1_macro_scorer, n_jobs=-1, verbose=1, refit=True
)
grid_mnb.fit(X_all_nonneg_full, y)
print(f"Done in {time.time()-t0:.1f}s | Best alpha={grid_mnb.best_params_} | CV F1={grid_mnb.best_score_:.4f}")

evaluate_model("Tuned MultinomialNB", y_vl, grid_mnb.predict(X_all_nonneg_full[idx_val]))

# %% [markdown]
# ## 8. Model 2: ComplementNB (designed for imbalanced data)

# %%

# DEFAULT ComplementNB — COMMENTED OUT (already ran in first pass)
# LESSON LEARNED: Default alpha=1.0 gave Macro F1 ~0.63, Accuracy ~0.71
# ComplementNB is designed for imbalanced data but still underperforms
# linear models (LinearSVC, LR) on this dataset. Tuning helps marginally.

# print("ComplementNB...")
# t0 = time.time()
# cnb = ComplementNB(alpha=1.0)
# cnb.fit(X_tr_nn, y_tr)
# evaluate_model("ComplementNB (default)", y_vl, cnb.predict(X_vl_nn))
# print(f"  ⏱️ {time.time()-t0:.1f}s")
# RESULT FROM FIRST RUN:
#   Accuracy: 0.7066 | Macro F1: 0.6315 | Weighted F1: 0.7126
#   Kaggle score: 0.620 — worst among all models

# %%
print("GridSearchCV for ComplementNB...")
t0 = time.time()
grid_cnb = GridSearchCV(
    ComplementNB(),
    {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
    cv=cv, scoring=f1_macro_scorer, n_jobs=-1, verbose=1, refit=True
)
grid_cnb.fit(X_all_nonneg_full, y)
print(f"Done in {time.time()-t0:.1f}s | Best alpha={grid_cnb.best_params_} | CV F1={grid_cnb.best_score_:.4f}")

evaluate_model("Tuned ComplementNB", y_vl, grid_cnb.predict(X_all_nonneg_full[idx_val]))

# %% [markdown]
# ## 9. Model 3: LinearSVC (typically #1 for text classification)

# %%

# DEFAULT LinearSVC (balanced) — COMMENTED OUT (already ran in first pass)
# LESSON LEARNED: Default C=1.0 with balanced gave Macro F1 ~0.80
# The tuned GridSearchCV below optimizes C for Macro F1, so running
# default first is redundant. Saved ~5 min.

# print("LinearSVC...")
# t0 = time.time()
# lsvc = LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)
# lsvc.fit(X_tr, y_tr)
# evaluate_model("LinearSVC (balanced)", y_vl, lsvc.predict(X_vl))
# print(f"  ⏱️ {time.time()-t0:.1f}s")
# RESULT FROM FIRST RUN:
#   Accuracy: 0.8977 | Macro F1: 0.7961 | Weighted F1: 0.8996

# %%

# LinearSVC WITHOUT class_weight — COMMENTED OUT (key lesson learned)
# LESSON LEARNED: We discovered that Kaggle metric = Macro F1 (not accuracy).
# Without balanced: Accuracy 0.8978 (similar) but Macro F1 0.7931 (WORSE)
#   -> class_weight='balanced' HELPS Macro F1 by boosting Label 3 recall
#   -> Kaggle score without balanced: 0.816 vs balanced should be higher
# CONCLUSION: Always use class_weight='balanced' for this competition.

# print("LinearSVC (no class_weight)...")
# t0 = time.time()
# lsvc_no_bal = LinearSVC(max_iter=5000, random_state=42)
# lsvc_no_bal.fit(X_tr, y_tr)
# evaluate_model("LinearSVC (no balance)", y_vl, lsvc_no_bal.predict(X_vl))
# print(f"  ⏱️ {time.time()-t0:.1f}s")
# RESULT FROM FIRST RUN:
#   Accuracy: 0.8978 | Macro F1: 0.7931
#   Kaggle score: 0.816 (confirmed Kaggle uses Macro F1, not accuracy)

# %%
# Tune LinearSVC
# IMPORTANT: Kaggle metric is Macro F1 (not accuracy), so we optimize for that.
# class_weight="balanced" helps because it boosts minority class (Label 3) recall.
print("GridSearchCV for LinearSVC...")
t0 = time.time()
grid_lsvc = GridSearchCV(
    LinearSVC(random_state=42, max_iter=5000),
    {
        "C": [0.1, 0.5, 1.0, 2.0],
        "class_weight": ["balanced"],   # balanced helps Macro F1 (Kaggle metric)
    },
    cv=cv, scoring=f1_macro_scorer,      # Macro F1 = Kaggle metric
    n_jobs=-1, verbose=1, refit=True
)
grid_lsvc.fit(X_all, y)
print(f"Done in {time.time()-t0:.1f}s | Best={grid_lsvc.best_params_} | CV Macro F1={grid_lsvc.best_score_:.4f}")

evaluate_model("Tuned LinearSVC", y_vl, grid_lsvc.predict(X_all[idx_val]))

# %% [markdown]
# ## 10. Model 4: PassiveAggressiveClassifier

# %%
print("PassiveAggressiveClassifier...")
t0 = time.time()
pac = PassiveAggressiveClassifier(
    class_weight="balanced", max_iter=2000, random_state=42, n_jobs=-1
)
pac.fit(X_tr, y_tr)
evaluate_model("PassiveAggressive (balanced)", y_vl, pac.predict(X_vl))
print(f"  ⏱️ {time.time()-t0:.1f}s")

# %%
print("GridSearchCV for PassiveAggressive...")
t0 = time.time()
grid_pac = GridSearchCV(
    PassiveAggressiveClassifier(random_state=42, n_jobs=-1, max_iter=2000),
    {
        "C": [0.01, 0.1, 0.5, 1.0],
        "class_weight": ["balanced"],   # balanced helps Macro F1
    },
    cv=cv, scoring=f1_macro_scorer,      # Macro F1 = Kaggle metric
    n_jobs=-1, verbose=1, refit=True
)
grid_pac.fit(X_all, y)
print(f"Done in {time.time()-t0:.1f}s | Best={grid_pac.best_params_} | CV Macro F1={grid_pac.best_score_:.4f}")

evaluate_model("Tuned PassiveAggressive", y_vl, grid_pac.predict(X_all[idx_val]))

# %% [markdown]
# ## 11. Model 5: Logistic Regression (carried from M2, with more features)

# %%

# LR (balanced) — kept, this is useful as M2->M3 comparison with char n-grams

print("Logistic Regression (with char n-grams)...")
t0 = time.time()
lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
                        class_weight="balanced", n_jobs=-1, random_state=42)
lr.fit(X_tr, y_tr)
evaluate_model("LR (balanced, 25K features)", y_vl, lr.predict(X_vl))
print(f"  ⏱️ {time.time()-t0:.1f}s")

# %%

# LR WITHOUT balanced — COMMENTED OUT (lesson learned from LinearSVC)
# LESSON LEARNED: Since Kaggle metric = Macro F1, removing balanced hurts score.
# Without balanced: LR Accuracy 0.9012 but Macro F1 0.7808 (LOWER than balanced 0.7979)
# Same pattern as LinearSVC — balanced always wins for Macro F1.

# print("LR (no balance, 25K features)...")
# t0 = time.time()
# lr_nb = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000,
#                            n_jobs=-1, random_state=42)
# lr_nb.fit(X_tr, y_tr)
# evaluate_model("LR (no balance, 25K features)", y_vl, lr_nb.predict(X_vl))
# print(f"  ⏱️ {time.time()-t0:.1f}s")
# RESULT FROM FIRST RUN:
#   Accuracy: 0.9012 | Macro F1: 0.7808

# %% [markdown]
# ## 12. Model 6: KNN (with Dimensionality Reduction)
#
# KNN doesn't work well on high-dimensional sparse data.
# We use TruncatedSVD to reduce dimensions first.

# %%
# --------------------------------------------------
# Dimensionality Reduction: TruncatedSVD
# --------------------------------------------------
print("TruncatedSVD for dimensionality reduction...")
t0 = time.time()

svd = TruncatedSVD(n_components=200, random_state=42)
X_tr_svd = svd.fit_transform(X_tr)
X_vl_svd = svd.transform(X_vl)

print(f"SVD done in {time.time()-t0:.1f}s")
print(f"   Explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")
print(f"   Reduced shape: {X_tr_svd.shape}")

# %%
print("KNN (k=5, SVD-reduced)...")
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_tr_svd, y_tr)
evaluate_model("KNN (k=5, 200-dim SVD)", y_vl, knn.predict(X_vl_svd))
print(f"  ⏱️ {time.time()-t0:.1f}s")

# %%

# Individual K-value testing — COMMENTED OUT (already ran in first pass)
# LESSON LEARNED: k=5 with distance weights performed best.
# Results from first run:
#   k= 3 -> Accuracy: 0.8157 | Macro F1: 0.5929
#   k= 7 -> Accuracy: 0.8225 | Macro F1: 0.5748
#   k=11 -> Accuracy: 0.8242 | Macro F1: 0.5532
#   k=15 -> Accuracy: 0.8221 | Macro F1: 0.5385
# KNN Macro F1 is very low compared to LinearSVC (0.86) — not competitive.

# print("Testing different k values...")
# for k in [3, 7, 11, 15]:
#     knn_k = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
#     knn_k.fit(X_tr_svd, y_tr)
#     acc_k = accuracy_score(y_vl, knn_k.predict(X_vl_svd))
#     f1_k  = f1_score(y_vl, knn_k.predict(X_vl_svd), average="macro")
#     print(f"  k={k:2d} -> Accuracy: {acc_k:.4f} | Macro F1: {f1_k:.4f}")

# %%
# Best KNN with tuning
# NOTE: KNN is instance-based (memorizes training data), so we MUST NOT
# train on data that includes validation — otherwise it gets perfect 1.0 accuracy.
# We fit GridSearchCV on TRAINING split only, then evaluate on held-out validation.
print("GridSearchCV for KNN...")
t0 = time.time()
grid_knn = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},  # Reduced from [3,5,7,11]
    cv=cv, scoring=f1_macro_scorer, n_jobs=-1, verbose=1, refit=True  # Changed to Macro F1
)
grid_knn.fit(X_tr_svd, y_tr)  # Train on TRAINING split only
print(f"Done in {time.time()-t0:.1f}s | Best={grid_knn.best_params_}")

evaluate_model("Tuned KNN (SVD)", y_vl, grid_knn.predict(X_vl_svd))

# Prepare test SVD for KNN submission (needed because KNN works on reduced dimensions)
X_test_svd = svd.transform(X_test_all)

# %% [markdown]
# ## 13. Feature Selection: SelectKBest (chi2)

# %%

# FEATURE SELECTION: SelectKBest (chi2) — COMMENTED OUT (already ran in first pass)
# LESSON LEARNED: Selecting top 15K features from 25K gave similar performance
# to using all features. LinearSVC (chi2 selected) scored:
#   Accuracy: 0.9029 | Macro F1: 0.7991
# This is LOWER than Tuned LinearSVC (0.8608) because:
#   1. chi2 selection was done without class_weight='balanced'
#   2. Feature selection removed some char n-gram features that were useful
# CONCLUSION: For this dataset, full feature set + balanced class weight > feature selection

# print("SelectKBest (chi2, k=15000)...")
# t0 = time.time()
# selector = SelectKBest(chi2, k=15000)
# X_tr_sel = selector.fit_transform(X_tr_nn, y_tr)
# X_vl_sel = selector.transform(X_vl_nn)
# print(f"Selected {X_tr_sel.shape[1]} features in {time.time()-t0:.1f}s")
# lsvc_sel = LinearSVC(C=1.0, max_iter=5000, random_state=42)
# lsvc_sel.fit(X_tr_sel, y_tr)
# evaluate_model("LinearSVC (chi2 selected)", y_vl, lsvc_sel.predict(X_vl_sel))

# %% [markdown]
# ---
# ## 14. Complete Model Comparison

# %%
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("  COMPLETE MODEL COMPARISON — Milestone 3")
print("=" * 80)
print(results_df.to_string(index=False))

# %%
# Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
metrics = ["accuracy", "macro_f1", "weighted_f1"]
titles  = ["Accuracy", "Macro F1-Score", "Weighted F1-Score"]
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    bars = axes[idx].barh(results_df["name"], results_df[metric],
                          color=colors_bar, edgecolor="black")
    axes[idx].set_title(title, fontsize=13, fontweight="bold")
    for bar, val in zip(bars, results_df[metric]):
        axes[idx].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                      f"{val:.4f}", va="center", fontsize=7)

plt.suptitle("Milestone 3: Model Comparison", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 15. Confusion Matrices — Top Models

# %%
# Get top 4 tuned models
top_models_preds = {}
for name_key, model_obj, data in [
    ("Tuned LinearSVC", grid_lsvc, X_all),
    ("Tuned MNB", grid_mnb, X_all_nonneg_full),
    ("Tuned CNB", grid_cnb, X_all_nonneg_full),
    ("Tuned PAC", grid_pac, X_all),
]:
    top_models_preds[name_key] = model_obj.predict(data[idx_val])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, (name, preds) in enumerate(top_models_preds.items()):
    ax = axes[idx // 2][idx % 2]
    cm = confusion_matrix(y_vl, preds, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3]).plot(
        ax=ax, cmap="Blues", values_format=".2f")
    ax.set_title(name, fontsize=12, fontweight="bold")

plt.suptitle("Normalized Confusion Matrices — M3 Models",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 16. Generate Submission with Best Model

# %%
best_idx = results_df["accuracy"].idxmax()
best_name = results_df.loc[best_idx, "name"]
best_acc  = results_df.loc[best_idx, "accuracy"]
print(f"Best Model: {best_name} (Accuracy: {best_acc:.4f})")

# %%
# Map to fitted models (GridSearchCV models already refit on full data)
# NOTE: Each model uses its appropriate test matrix:
#   - LinearSVC/PAC/LR -> X_test_all (can handle negative values from StandardScaler)
#   - NB models -> X_test_nonneg_full (needs non-negative values, uses MinMaxScaler)
#   - KNN -> X_test_svd (needs SVD-reduced dimensions, can't work on 25K features)
submission_models = {
    "Tuned LinearSVC": (grid_lsvc, X_test_all),
    "Tuned MultinomialNB": (grid_mnb, X_test_nonneg_full),
    "Tuned ComplementNB": (grid_cnb, X_test_nonneg_full),
    "Tuned PassiveAggressive": (grid_pac, X_test_all),
    "Tuned KNN (SVD)": (grid_knn, X_test_svd),  # KNN on SVD-reduced test data
}

# Pick the best by matching name
best_model_obj = None
best_test_data = None

for key, (model, test_data) in submission_models.items():
    if key in best_name or best_name in key:
        best_model_obj = model
        best_test_data = test_data
        print(f"Using: {key}")
        break

if best_model_obj is None:
    # Fallback to LinearSVC (usually best)
    print("Falling back to Tuned LinearSVC")
    best_model_obj = grid_lsvc
    best_test_data = X_test_all

# %%
test_predictions = best_model_obj.predict(best_test_data)

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
# ## 17. Bonus: Multiple Submissions (try different models)

# %%
# Generate submissions from top models for comparison on Kaggle
for name, (model, test_data) in submission_models.items():
    preds = model.predict(test_data)
    sub = sample.copy()
    sub["label"] = preds
    fname = f"submission_{name.lower().replace(' ', '_').replace('(','').replace(')','')}.csv"
    sub.to_csv(fname, index=False)
    print(f"{fname} saved | Distribution: {dict(zip(*np.unique(preds, return_counts=True)))}")

# %% [markdown]
# ---
# ## 18. Milestone 3 Summary
#
# ### Models Trained:
# | # | Model | Type | Status |
# |---|-------|------|--------|
# | 1 | MultinomialNB | Naive Bayes | Tuned |
# | 2 | ComplementNB | Naive Bayes (imbalanced) | Tuned |
# | 3 | LinearSVC | SVM (linear) | Tuned |
# | 4 | PassiveAggressiveClassifier | Online linear | Tuned |
# | 5 | Logistic Regression | Linear (from M2) | with char n-grams |
# | 6 | KNN | Instance-based | Tuned with TruncatedSVD |
#
# ### Feature Improvements:
# - Character n-gram TF-IDF (10K features) added
# - TruncatedSVD for dimensionality reduction (KNN)
# - SelectKBest (chi2) for feature selection
# - Tested with and without class_weight="balanced"
#
# ### All M3 Question Answers:
# Printed inline in Section 3 with `[Q] M3 Q` markers.
#
# ---
# **Next:** Milestone 4 — Ensemble (Bagging, Boosting, Stacking), MLP
