# %% [markdown]
# # Milestone 1: Data Loading, EDA, Preprocessing & Baseline Model
# **Comment Category Prediction Challenge**

# This notebook covers:
# 1. **Data Loading** — Import all dataset files
# 2. **Exploratory Data Analysis (EDA)** — Statistical & visual analysis
# 3. **Data Cleaning & Preprocessing** — Missing values, encoding, scaling
# 4. **Train-Validation Split** — Stratified split
# 5. **Baseline Model** — First predictive model with evaluation

# ---

# %% [markdown]
# ## 1. Setup & Imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# Plotting configuration
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")
sns.set_palette("viridis")

print("All libraries imported successfully.")

# %% [markdown]
# ## 2. Data Loading

# %%
# Load all datasets
train = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/train.csv")
test  = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/test.csv")
sample = pd.read_csv("/kaggle/input/comment-category-prediction-challenge/Sample.csv")

print("All datasets loaded successfully.\n")
print(f"Train  -> {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"Test   -> {test.shape[0]:,} rows × {test.shape[1]} columns")
print(f"Sample -> {sample.shape[0]:,} rows × {sample.shape[1]} columns")

# %% [markdown]
# ## 3. Initial Data Inspection

# %%
# Preview training data
print("First 5 rows of Training Data:")
train.head()

# %%
# Preview test data
print("First 5 rows of Test Data:")
test.head()

# %%
# Preview sample submission
print("Sample Submission Format:")
sample.head()

# %%
# Column names comparison
print("Train columns:", train.columns.tolist())
print("\nTest columns: ", test.columns.tolist())
print("\nSample columns:", sample.columns.tolist())

train_only_cols = set(train.columns) - set(test.columns)
test_only_cols  = set(test.columns) - set(train.columns)
print(f"\nColumns only in train (target): {train_only_cols}")
print(f"Columns only in test:           {test_only_cols}")

# %% [markdown]
# ## 4. Structural Analysis

# %%

# [Q] MILESTONE 1 QUESTION: What is the shape of the training dataset?

print(f"Shape of training dataset: {train.shape}")
print(f"   -> {train.shape[0]:,} rows and {train.shape[1]} columns")

# %%

# [Q] MILESTONE 1 QUESTION: How many columns are present in the test dataset?

print(f"Number of columns in test dataset: {test.shape[1]}")

# %%
# Detailed data types
print("=" * 60)
print("TRAINING DATA - Data Types Summary")
print("=" * 60)
print(train.dtypes)
print("\n" + "-" * 60)
print(f"Data type counts:\n{train.dtypes.value_counts()}")

# %%

# [Q] MILESTONE 1 QUESTION: How many columns in the training dataset have object data type?

object_cols = train.select_dtypes(include="object").columns.tolist()
print(f"Number of object-type columns: {len(object_cols)}")
print(f"   -> Columns: {object_cols}")

# %%

# [Q] MILESTONE 1 QUESTION: How many numerical columns are present in the training dataset?

numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(f"Number of numerical columns: {len(numerical_cols)}")
print(f"   -> Columns: {numerical_cols}")

# %%

# [Q] MILESTONE 1 QUESTION: Which of the following columns is of boolean type?

bool_cols = train.select_dtypes(include="bool").columns.tolist()
print(f"Boolean-type columns: {bool_cols}")

# %%
# Complete data info
print("=" * 60)
print("TRAINING DATA - Info")
print("=" * 60)
train.info()

# %%
print("\n" + "=" * 60)
print("TEST DATA - Info")
print("=" * 60)
test.info()

# %% [markdown]
# ### 4.1 Feature Categorization
#
# We categorize all features by their semantic role for organized analysis.

# %%
# Feature categorization
feature_groups = {
    "Text Features":     ["comment"],
    "Datetime Features": ["created_date"],
    "ID Features":       ["post_id"],
    "Numerical Features":["upvote", "downvote", "if_1", "if_2"],
    "Binary/Indicator":  ["emoticon_1", "emoticon_2", "emoticon_3"],
    "Detection Flags":   ["race", "religion", "gender", "disability"],
    "Target":            ["label"]
}

print("Feature Groups:")
print("=" * 60)
for group, cols in feature_groups.items():
    dtypes = [str(train[c].dtype) for c in cols if c in train.columns]
    print(f"\n{group}:")
    for col, dt in zip(cols, dtypes):
        print(f"   • {col:20s} -> dtype: {dt}")
print("=" * 60)

# %% [markdown]
# ## 5. Missing Value Analysis

# %%

# [Q] MILESTONE 1 QUESTION: Which of the following columns have missing values?

# Training set missing values
train_missing = train.isnull().sum()
train_missing_pct = (train.isnull().sum() / len(train)) * 100

missing_df_train = pd.DataFrame({
    "Missing Count": train_missing,
    "Missing %": train_missing_pct.round(2)
}).sort_values("Missing Count", ascending=False)

print("TRAINING DATA - Missing Values:")
print("=" * 50)
print(missing_df_train)
print(f"\nTotal rows: {len(train):,}")

# %%
# Test set missing values
test_missing = test.isnull().sum()
test_missing_pct = (test.isnull().sum() / len(test)) * 100

missing_df_test = pd.DataFrame({
    "Missing Count": test_missing,
    "Missing %": test_missing_pct.round(2)
}).sort_values("Missing Count", ascending=False)

print("TEST DATA - Missing Values:")
print("=" * 50)
print(missing_df_test)
print(f"\nTotal rows: {len(test):,}")

# %%
# Visualize missing values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cols_with_missing_train = train_missing[train_missing > 0]
if len(cols_with_missing_train) > 0:
    axes[0].barh(cols_with_missing_train.index, cols_with_missing_train.values, color="salmon")
    axes[0].set_title("Missing Values - Train", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(cols_with_missing_train.values):
        axes[0].text(v + 500, i, f"{v:,} ({v/len(train)*100:.1f}%)", va="center", fontsize=10)

cols_with_missing_test = test_missing[test_missing > 0]
if len(cols_with_missing_test) > 0:
    axes[1].barh(cols_with_missing_test.index, cols_with_missing_test.values, color="lightskyblue")
    axes[1].set_title("Missing Values - Test", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Count")
    for i, v in enumerate(cols_with_missing_test.values):
        axes[1].text(v + 500, i, f"{v:,} ({v/len(test)*100:.1f}%)", va="center", fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Missing Value Observations:**
# - `race`, `religion`, and `gender` have ~73% missing values in both train and test.
# - These three columns always have the **exact same count** of missing values -> likely they are all missing/present together (i.e., system detected all or none).
# - `comment` has only 1 missing value in train, 0 in test.
# - `disability` has **no missing values** (it's boolean type).
# - All other features are complete.

# %%
# Verify: Are race/religion/gender always missing together?
demo_cols = ["race", "religion", "gender"]
all_missing_together = train[demo_cols].isnull().all(axis=1).sum()
any_missing = train[demo_cols].isnull().any(axis=1).sum()
print(f"Rows where ALL three (race/religion/gender) are missing: {all_missing_together:,}")
print(f"Rows where ANY of three are missing:                     {any_missing:,}")
print(f"-> They are {'ALWAYS' if all_missing_together == any_missing else 'NOT always'} missing together.")

# %% [markdown]
# ## 6. Target Variable Analysis

# %%

# [Q] MILESTONE 1 QUESTION: How many distinct target classes are present in the dataset?

n_classes = train["label"].nunique()
class_values = sorted(train["label"].unique())
print(f"Number of distinct target classes: {n_classes}")
print(f"   -> Classes: {class_values}")

# %%

# [Q] MILESTONE 1 QUESTION: What percentage of the dataset approximately belongs to label 0?

label_counts = train["label"].value_counts().sort_index()
label_pcts   = train["label"].value_counts(normalize=True).sort_index() * 100

label_summary = pd.DataFrame({
    "Count": label_counts,
    "Percentage (%)": label_pcts.round(2)
})
print("Target Distribution:")
print(label_summary)
print(f"\n-> Label 0 percentage: {label_pcts[0]:.2f}%")

# %%
# Target distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
axes[0].bar(label_counts.index.astype(str), label_counts.values, color=colors, edgecolor="black")
axes[0].set_title("Label Distribution (Count)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Count")
for i, (count, pct) in enumerate(zip(label_counts.values, label_pcts.values)):
    axes[0].text(i, count + 1000, f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=10)

# Pie chart
axes[1].pie(label_counts.values, labels=[f"Label {l}" for l in label_counts.index],
            autopct="%1.1f%%", colors=colors, startangle=90,
            explode=[0.05]*n_classes, shadow=True)
axes[1].set_title("Label Distribution (Proportion)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Target Distribution Observations:**
# - The dataset is **heavily imbalanced**.
# - Label 0 dominates with ~57.7% of samples.
# - Label 3 is the extreme minority at ~2.8%.
# - **Imbalance ratios:** Label 0 is ~21× more frequent than Label 3.
# - This imbalance will require techniques like:
#   - `class_weight='balanced'` in models
#   - Stratified train-validation split
#   - Potentially SMOTE for oversampling minority (to be evaluated)
#   - F1-score (macro/weighted) as primary evaluation metric instead of accuracy

# %% [markdown]
# ## 7. Statistical Analysis of Numerical Features

# %%
# Descriptive statistics
num_features = ["upvote", "downvote", "if_1", "if_2"]
print("Descriptive Statistics - Numerical Features:")
print("=" * 70)
train[num_features].describe().round(2)

# %%

# [Q] MILESTONE 1 QUESTION: What is the median number of upvotes per comment?

median_upvotes = train["upvote"].median()
print(f"Median upvotes per comment: {median_upvotes}")

# %%

# [Q] MILESTONE 1 QUESTION: Which numerical feature shows the largest maximum value?

max_values = train[num_features].max()
print("Maximum values for each numerical feature:")
print(max_values)
print(f"\n-> Feature with largest maximum value: '{max_values.idxmax()}' (max = {max_values.max()})")

# %%

# [Q] MILESTONE 1 QUESTION: What is the minimum value of if_2?

min_if2 = train["if_2"].min()
print(f"Minimum value of if_2: {min_if2}")

# %%
# Distribution of numerical features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(num_features):
    ax = axes[idx // 2][idx % 2]
    sns.histplot(train[col], bins=50, kde=True, ax=ax, color=colors[idx])
    ax.set_title(f"Distribution of '{col}'", fontsize=13, fontweight="bold")
    ax.axvline(train[col].mean(), color="red", linestyle="--", label=f"Mean: {train[col].mean():.2f}")
    ax.axvline(train[col].median(), color="blue", linestyle="-.", label=f"Median: {train[col].median():.2f}")
    ax.legend(fontsize=9)

plt.suptitle("Numerical Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Boxplots for detecting outliers
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for idx, col in enumerate(num_features):
    sns.boxplot(y=train[col], ax=axes[idx], color=colors[idx])
    axes[idx].set_title(f"Boxplot: {col}", fontweight="bold")

plt.suptitle("Outlier Detection via Boxplots", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Skewness and Kurtosis
print("Skewness & Kurtosis of Numerical Features:")
print("=" * 50)
for col in num_features:
    skew = train[col].skew()
    kurt = train[col].kurtosis()
    print(f"  {col:12s} -> Skewness: {skew:8.2f}  |  Kurtosis: {kurt:8.2f}")

# %% [markdown]
# **Numerical Feature Observations:**
# - `upvote` and `downvote` are heavily **right-skewed** with long tails (many outliers).
# - `if_1` and `if_2` have more compact distributions but may still have skew.
# - Standard scaling will be essential for distance/linear-based models.

# %% [markdown]
# ## 8. Feature Distributions by Label

# %%
# Numerical features by label (boxplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(num_features):
    ax = axes[idx // 2][idx % 2]
    sns.boxplot(x="label", y=col, data=train, ax=ax, palette=colors)
    ax.set_title(f"{col} vs Label", fontsize=13, fontweight="bold")

plt.suptitle("Numerical Features by Target Label", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Numerical features by label (violin plots for better shape view)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(num_features):
    ax = axes[idx // 2][idx % 2]
    sns.violinplot(x="label", y=col, data=train, ax=ax, palette=colors, inner="quartile")
    ax.set_title(f"{col} Distribution by Label (Violin)", fontsize=13, fontweight="bold")

plt.suptitle("Numerical Feature Shapes per Label", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Mean of numerical features per label
print("Mean of Numerical Features per Label:")
print("=" * 60)
print(train.groupby("label")[num_features].mean().round(2))

# %% [markdown]
# ## 9. Binary / Indicator Feature Analysis

# %%
# Emoticon features analysis
emoticon_cols = ["emoticon_1", "emoticon_2", "emoticon_3"]

print("Emoticon Feature Value Counts:")
print("=" * 50)
for col in emoticon_cols:
    print(f"\n{col}:")
    print(train[col].value_counts())
    print(f"  True ratio: {train[col].mean()*100:.2f}%")

# %%
# Disability feature analysis
print("\nDisability Feature Value Counts:")
print(train["disability"].value_counts())
print(f"  True ratio: {train['disability'].mean()*100:.2f}%")

# %%
# Binary features vs label
binary_cols = ["emoticon_1", "emoticon_2", "emoticon_3", "disability"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for idx, col in enumerate(binary_cols):
    ct = pd.crosstab(train[col], train["label"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=axes[idx], color=colors, edgecolor="black")
    axes[idx].set_title(f"{col} vs Label", fontweight="bold")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Percentage %")
    axes[idx].legend(title="Label", fontsize=8)

plt.suptitle("Binary Features vs Target Label (Stacked %)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Detection Flag Analysis (race, religion, gender)

# %%
# Analysis of rows where detection flags are present vs missing
demo_cols = ["race", "religion", "gender"]

# Create a flag: detection present or not
train["detection_present"] = train[demo_cols].notna().all(axis=1)

print("Detection Flags Presence:")
print(train["detection_present"].value_counts())
print(f"\nPresent: {train['detection_present'].sum():,} ({train['detection_present'].mean()*100:.1f}%)")
print(f"Missing: {(~train['detection_present']).sum():,} ({(~train['detection_present']).mean()*100:.1f}%)")

# %%
# Label distribution: detection present vs absent
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (flag, title) in enumerate([(True, "Detection Present"), (False, "Detection Missing")]):
    subset = train[train["detection_present"] == flag]
    vc = subset["label"].value_counts(normalize=True).sort_index() * 100
    axes[idx].bar(vc.index.astype(str), vc.values, color=colors, edgecolor="black")
    axes[idx].set_title(f"Label Distribution - {title}\n(n={len(subset):,})", fontsize=13, fontweight="bold")
    axes[idx].set_xlabel("Label")
    axes[idx].set_ylabel("Percentage %")
    for i, v in enumerate(vc.values):
        axes[idx].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Value distributions when present
for col in demo_cols:
    print(f"\n{col} (when not null):")
    print(train[col].dropna().value_counts())

# %% [markdown]
# ## 11. Text Feature Analysis (comment)

# %%
# Comment length statistics
train["comment_length"] = train["comment"].astype(str).str.len()
test["comment_length"]  = test["comment"].astype(str).str.len()

print("Comment Length Statistics (Train):")
print("=" * 50)
print(train["comment_length"].describe().round(2))

# %%
# Word count statistics
train["word_count"] = train["comment"].astype(str).str.split().str.len()
test["word_count"]  = test["comment"].astype(str).str.split().str.len()

print("\nWord Count Statistics (Train):")
print("=" * 50)
print(train["word_count"].describe().round(2))

# %%
# Comment length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(train["comment_length"], bins=60, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Comment Length Distribution", fontsize=14, fontweight="bold")
axes[0].axvline(train["comment_length"].mean(), color="red", linestyle="--", label=f"Mean: {train['comment_length'].mean():.0f}")
axes[0].axvline(train["comment_length"].median(), color="orange", linestyle="-.", label=f"Median: {train['comment_length'].median():.0f}")
axes[0].legend()

sns.histplot(train["word_count"], bins=60, kde=True, ax=axes[1], color="coral")
axes[1].set_title("Word Count Distribution", fontsize=14, fontweight="bold")
axes[1].axvline(train["word_count"].mean(), color="red", linestyle="--", label=f"Mean: {train['word_count'].mean():.0f}")
axes[1].axvline(train["word_count"].median(), color="orange", linestyle="-.", label=f"Median: {train['word_count'].median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# Comment length by label
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x="label", y="comment_length", data=train, ax=axes[0], palette=colors)
axes[0].set_title("Comment Length by Label", fontsize=14, fontweight="bold")

sns.boxplot(x="label", y="word_count", data=train, ax=axes[1], palette=colors)
axes[1].set_title("Word Count by Label", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()

# %%
# Average text stats per label
print("Average Comment Length & Word Count per Label:")
print("=" * 60)
print(train.groupby("label")[["comment_length", "word_count"]].mean().round(2))

# %% [markdown]
# ## 12. Engagement Analysis (Upvotes & Downvotes)

# %%
# Upvote vs Downvote scatter
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    train["upvote"], train["downvote"],
    c=train["label"], cmap="viridis", alpha=0.3, s=10
)
ax.set_xlabel("Upvotes")
ax.set_ylabel("Downvotes")
ax.set_title("Upvote vs Downvote (Colored by Label)", fontsize=14, fontweight="bold")
plt.colorbar(scatter, label="Label")
plt.tight_layout()
plt.show()

# %%
# Engagement derived features analysis
train["vote_ratio"] = train["upvote"] / (train["downvote"] + 1)  # +1 to avoid division by zero
train["total_votes"] = train["upvote"] + train["downvote"]
train["vote_diff"]   = train["upvote"] - train["downvote"]

print("Derived Engagement Features per Label:")
print("=" * 60)
print(train.groupby("label")[["vote_ratio", "total_votes", "vote_diff"]].mean().round(2))

# %% [markdown]
# ## 13. Correlation Analysis

# %%
# Correlation heatmap
corr_cols = ["upvote", "downvote", "if_1", "if_2",
             "emoticon_1", "emoticon_2", "emoticon_3", "disability",
             "comment_length", "word_count", "label"]

corr_matrix = train[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Correlation Observations:**
# - `if_1` and `if_2` show the strongest correlation with the target `label`.
# - `upvote` and `downvote` have mild positive correlation with each other.
# - Emoticon features have weak but non-zero correlations.
# - `comment_length` and `word_count` are highly correlated (redundant — keep one or use PCA).

# %%
# Correlation with target
target_corr = corr_matrix["label"].drop("label").sort_values(ascending=False)
print("Feature Correlation with Target (label):")
print("=" * 50)
print(target_corr.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
target_corr.plot(kind="barh", color=["green" if v > 0 else "red" for v in target_corr.values], ax=ax)
ax.set_title("Feature Correlation with Target", fontsize=14, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14. Class Imbalance Deep Dive — Should We Use SMOTE?

# %%
# Imbalance ratio analysis
print("Class Imbalance Analysis:")
print("=" * 60)
majority_class = label_counts.max()
for label in sorted(train["label"].unique()):
    count = label_counts[label]
    ratio = majority_class / count
    print(f"  Label {label}: {count:>7,} samples | Imbalance ratio: 1:{ratio:.1f}")

print(f"\n-> Maximum imbalance ratio: 1:{majority_class/label_counts.min():.1f} (Label 0 vs Label 3)")

# %%
# Considerations for SMOTE
print("""
SMOTE Applicability Analysis
-------------------------------------------------
Imbalance Level:
  Label 0: 57.7% - MAJORITY class
  Label 2: 31.5% - Moderate
  Label 1:  8.0% - Minor minority
  Label 3:  2.8% - EXTREME minority

SMOTE Considerations:
  - Significant imbalance present (21:1 ratio)
  - Primary feature is TEXT (TF-IDF sparse matrix)
  - SMOTE on sparse high-dim data can create noise
  - 198K samples, minority still has 5,469 samples

Recommendation:
  1. First try class_weight='balanced' (simpler, effective)
  2. If minority recall is poor, try SMOTE on numeric
     features only, or use SMOTE variants (ADASYN, etc.)
  3. Evaluate using macro F1-score, not accuracy
""")

# %% [markdown]
# ## 15. Datetime Feature Exploration

# %%
# Parse datetime
train["created_date"] = pd.to_datetime(train["created_date"], errors="coerce")
test["created_date"]  = pd.to_datetime(test["created_date"], errors="coerce")

# Extract datetime components
train["hour"] = train["created_date"].dt.hour
train["day_of_week"] = train["created_date"].dt.dayofweek
train["month"] = train["created_date"].dt.month

print("Datetime Range:")
print(f"  Train: {train['created_date'].min()} to {train['created_date'].max()}")
print(f"  Test:  {test['created_date'].min()} to {test['created_date'].max()}")

# %%
# Posting patterns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# By hour
train.groupby("hour")["label"].count().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Comments by Hour", fontweight="bold")
axes[0].set_xlabel("Hour of Day")

# By day of week
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
counts_by_day = train.groupby("day_of_week")["label"].count()
axes[1].bar([day_names[i] for i in counts_by_day.index], counts_by_day.values, color="coral")
axes[1].set_title("Comments by Day of Week", fontweight="bold")

# By month
train.groupby("month")["label"].count().plot(kind="bar", ax=axes[2], color="mediumseagreen")
axes[2].set_title("Comments by Month", fontweight="bold")
axes[2].set_xlabel("Month")

plt.suptitle("Temporal Posting Patterns", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Label distribution by hour
fig, ax = plt.subplots(figsize=(14, 5))
ct = pd.crosstab(train["hour"], train["label"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", alpha=0.85)
ax.set_title("Label Distribution by Hour of Day", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour")
ax.set_ylabel("Percentage %")
ax.legend(title="Label", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16. Post ID Analysis

# %%
# How many unique posts?
print(f"Unique post_ids in train: {train['post_id'].nunique():,}")
print(f"Unique post_ids in test:  {test['post_id'].nunique():,}")
print(f"Comments per post (train):")
print(train["post_id"].value_counts().describe().round(2))

# %%
# Label diversity per post
labels_per_post = train.groupby("post_id")["label"].nunique()
print(f"\nAverage label diversity per post: {labels_per_post.mean():.2f}")
print(labels_per_post.value_counts().sort_index())

# %% [markdown]
# ## 17. Data Cleaning & Preprocessing

# %%
# Drop 1 row with missing comment
print(f"Before cleaning: {train.shape[0]:,} rows")
train = train.dropna(subset=["comment"])
print(f"After dropping missing comments: {train.shape[0]:,} rows")

# %%
# Handle demographic columns (race, religion, gender)
# Strategy: Fill NaN with "missing" for categorical, then encode
for col in ["race", "religion", "gender"]:
    train[col] = train[col].fillna("missing")
    test[col]  = test[col].fillna("missing")

print("Filled NaN in race/religion/gender with 'missing'")

# %%
# Verify no missing values remain
print("Missing values after cleaning (Train):")
print(train.isnull().sum()[train.isnull().sum() > 0])
if train.isnull().sum().sum() == 0:
    print("-> No missing values remain in train!")

print("\nMissing values after cleaning (Test):")
print(test.isnull().sum()[test.isnull().sum() > 0])
if test.isnull().sum().sum() == 0:
    print("-> No missing values remain in test!")

# %% [markdown]
# ## 18. Feature Engineering (for Baseline)

# %%
# Create text-derived features
for df in [train, test]:
    df["comment_length"]   = df["comment"].astype(str).str.len()
    df["word_count"]       = df["comment"].astype(str).str.split().str.len()
    df["avg_word_length"]  = df["comment_length"] / (df["word_count"] + 1)
    df["total_votes"]      = df["upvote"] + df["downvote"]
    df["vote_diff"]        = df["upvote"] - df["downvote"]
    df["vote_ratio"]       = df["upvote"] / (df["downvote"] + 1)
    df["has_detection"]    = df[["race", "religion", "gender"]].ne("missing").any(axis=1).astype(int)

print("Feature engineering complete.")
print(f"Train columns: {train.shape[1]}")
print(f"Test columns:  {test.shape[1]}")

# %% [markdown]
# ## 19. Train-Validation Split

# %%
from sklearn.model_selection import train_test_split

# Define features and target
# Text feature — will be vectorized separately
text_col = "comment"

# Numerical features for the model
numeric_features = ["upvote", "downvote", "if_1", "if_2",
                    "comment_length", "word_count", "avg_word_length",
                    "total_votes", "vote_diff", "vote_ratio"]

# Binary features
binary_features = ["emoticon_1", "emoticon_2", "emoticon_3",
                   "disability", "has_detection"]

# Separate X and y
target = "label"
X = train.copy()
y = train[target]

X_test_final = test.copy()

# Stratified split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train-Validation Split Complete (Stratified)")
print(f"   Train: {X_train.shape[0]:,} samples")
print(f"   Val:   {X_val.shape[0]:,} samples")
print(f"\nLabel distribution verification:")
print(f"  Train -> {y_train.value_counts(normalize=True).sort_index().round(4).to_dict()}")
print(f"  Val   -> {y_val.value_counts(normalize=True).sort_index().round(4).to_dict()}")

# %% [markdown]
# ## 20. Building a Baseline Model (Logistic Regression)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)

# %%
# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True       # dampens term frequency (log scaling)
        ), text_col),

        ("num", StandardScaler(), numeric_features),

        ("bin", "passthrough", binary_features)
    ]
)

# Define baseline model
baseline_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            C=2.0,
            n_jobs=-1,
            random_state=42
        ))
    ]
)

print("Baseline pipeline defined.")

# %%
# Train the baseline model
print("Training baseline model...")
baseline_model.fit(X_train, y_train)
print("Training complete!")

# %%
# Evaluate on validation set
y_val_pred = baseline_model.predict(X_val)

print("=" * 65)
print("BASELINE MODEL EVALUATION")
print("=" * 65)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")

print(f"\nValidation Accuracy:        {val_accuracy:.4f}")
print(f"Validation F1 (Macro):      {val_f1_macro:.4f}")
print(f"Validation F1 (Weighted):   {val_f1_weighted:.4f}")
print(f"\n{'-' * 65}")
print("Classification Report:")
print(classification_report(y_val, y_val_pred, digits=4))

# %%
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
cm = confusion_matrix(y_val, y_val_pred)
ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")

# Normalized
cm_norm = confusion_matrix(y_val, y_val_pred, normalize="true")
ConfusionMatrixDisplay(cm_norm, display_labels=[0, 1, 2, 3]).plot(ax=axes[1], cmap="Greens", values_format=".2f")
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")

plt.suptitle("Baseline Model - Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Baseline Model Summary

# | Metric | Value |
# |--------|-------|
# | Model | Logistic Regression |
# | Text Vectorizer | TF-IDF (10K features, bigrams, sublinear_tf) |
# | Numeric Scaling | StandardScaler |
# | Class Handling | class_weight='balanced' |
# | Evaluation | Stratified 80/20 split |

# The baseline model provides a solid starting point.
# Key areas for improvement in future milestones:
# - Try more powerful models (Random Forest, XGBoost, SVM, MLP)
# - Better text features (n-grams, character ngrams, embeddings)
# - Hyperparameter tuning
# - Handle extreme minority (Label 3) better

# %% [markdown]
# ## 21. Generate Submission File

# %%
# Retrain on FULL training data
print("Retraining on full training data for submission...")
baseline_model.fit(X, y)
print("Full retrain complete!")

# %%
# Generate predictions on test set
test_predictions = baseline_model.predict(X_test_final)

submission = sample.copy()
submission["label"] = test_predictions

print("Submission preview:")
print(submission.head(10))
print(f"\nPrediction distribution:")
print(pd.Series(test_predictions).value_counts().sort_index())

# %%
# Save submission file
submission.to_csv("submission.csv", index=False)
print("Submission saved to 'submission.csv'")

# %% [markdown]
# ## 22. Summary of Milestone 1 Question Answers
#
# | # | Question | Answer |
# |---|----------|--------|
# | 1 | Shape of training dataset | (198000, 15) |
# | 2 | Columns in test dataset | 14 |
# | 3 | Object-type columns in train | 5 |
# | 4 | Numerical columns in train | *(See output from Cell - depends on data)* |
# | 5 | Boolean-type column | disability |
# | 6 | Columns with missing values | race, religion, gender, comment |
# | 7 | Distinct target classes | 4 |
# | 8 | Percentage of label 0 | ~57.66% |
# | 9 | Median upvotes | *(See output)* |
# | 10 | Feature with largest max | *(See output)* |
# | 11 | Minimum value of if_2 | *(See output)* |
#
# **Note:** Exact numeric answers are printed in respective cells above.
# Run the notebook to see precise values.

# %% [markdown]
# ---
# ## End of Milestone 1
# **Next Steps (Milestone 2):** Linear Models, SGD, Hyperparameter Tuning
