# 📁 Data

The dataset for this project comes from a **Kaggle competition** and is not included in this repository due to file size constraints (train.csv = 73MB, test.csv = 37MB — exceeds GitHub's recommended limits).

---

## How to Get the Data

1. Go to the Kaggle competition: **Comment Category Prediction Challenge**
2. Download `train.csv`, `test.csv`, and `Sample.csv`
3. Place them in this `data/` folder:

```
data/
├── train.csv      ← 197,999 rows × 15 columns (includes label)
├── test.csv       ← 102,000 rows × 14 columns (no label)
└── Sample.csv     ← Submission format template
```

---

## Dataset Overview

| Property | Value |
|---|---|
| Train rows | 197,999 |
| Test rows | 102,000 |
| Features | 15 (text, engagement, demographic, datetime) |
| Target | `label` (4 classes: 0, 1, 2, 3) |
| Imbalance | 20.9:1 (Label 0: 57.7% vs Label 3: 2.7%) |

### Columns
| Column | Type | Description |
|---|---|---|
| `comment` | text | The online comment to classify |
| `created_date` | datetime | When the comment was posted |
| `upvote` | int | Number of upvotes |
| `downvote` | int | Number of downvotes |
| `emoticon_1/2/3` | float | Emoticon usage counts |
| `if_1`, `if_2` | float | Internal feature flags (heavily right-skewed, max=1860) |
| `race`, `religion`, `gender` | object | Demographic categories (~75-85% missing) |
| `disability` | bool | Disability-related flag |
| `post_id` | object | High-cardinality identifier (frequency-encoded) |
| `label` | int | Target: 0=neutral, 1=mild, 2=moderate, 3=severe |
