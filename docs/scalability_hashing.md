# Scaling Beyond RAM: HashingVectorizer + Out-of-Core SGD

The submitted model (see `src/milestone_5_final.py`) fits comfortably on 198k rows, but its design
has two scaling limits that showed up in this project as OOM crashes (M4 stacking with `n_jobs=-1`,
V4 pseudo-labeling on 260k × 60k):

| Limit | Why it breaks at scale |
|---|---|
| `TfidfVectorizer` vocabulary | Holds a Python `dict` of every n-gram seen before `max_features` pruning. With word 1–3-grams on 10M+ rows this alone can exceed RAM, and the whole corpus must be available to `fit`. |
| Batch solvers (`LinearSVC`, `lbfgs` LogReg, `StackingClassifier` CV) | Need the full sparse matrix in memory, and stacking multiplies that by base models × folds (and by workers when `n_jobs` > 1). |

## The alternative: stateless hashing + `partial_fit`

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

hv  = HashingVectorizer(n_features=2**20, ngram_range=(1, 2),
                        alternate_sign=False, norm="l2")
clf = SGDClassifier(loss="modified_huber", alpha=1e-5)   # or "log_loss" for probabilities

for chunk in pd.read_csv("train.csv", chunksize=50_000):   # constant memory
    X = hv.transform(chunk["comment"].map(clean_text))
    clf.partial_fit(X, chunk["label"], classes=[0, 1, 2, 3],
                    sample_weight=balanced_weight[chunk["label"]])
```

* **No vocabulary, no fit step.** A token is hashed straight to a column index, so memory is fixed by
  `n_features` regardless of corpus size, and train/test/serving use the identical mapping.
* **Streaming.** The corpus is read in chunks; the model sees each row `epochs` times.
* **Parallelisable.** Hashing is embarrassingly parallel across chunks/workers.

A runnable version is in [`src/hashing_sgd_experiment.py`](../src/hashing_sgd_experiment.py).

## What you give up (and how to mitigate it)

| Trade-off | Mitigation |
|---|---|
| **No IDF.** IDF needs global document frequencies. | Plain L2-normalised term frequency often loses little for linear models; or run one streaming pass to count document frequencies per hashed column, then apply `TfidfTransformer`-style weights. |
| **Hash collisions** (different n-grams share a column). | Increase `n_features` (2²² – 2²⁴ is common; the weight vector is only `n_features × n_classes` floats). Collision noise matters less than for tree models. |
| **Not invertible** — you can't map a column back to a word, which hurts error analysis. | Keep a small TF-IDF model on a sample for interpretation. |
| **`class_weight='balanced'` is unsupported in `partial_fit`.** | Compute balanced weights from one cheap pass over the label column and pass them as `sample_weight` (done in the script). |
| **Single model, no calibrated stacking.** | Stacking could be approximated by training several `partial_fit` models with different losses (hinge / log / PA) and fitting a small LogReg meta-model on a held-out stream. |
| **Sensitive to learning rate / epoch count.** | Shuffle chunks, use `learning_rate="optimal"` (default) or `adaptive`, and validate on a held-out slice of the stream. |

## How this would change the serving design at 10M+ rows

1. Replace `FeatureBuilder`'s two TF-IDF vectorizers with one `HashingVectorizer` (word) + one (char);
   keep the 19 structured features, whose `StandardScaler` can be fit with `partial_fit` too.
2. Train with chunked `partial_fit`; checkpoint the `SGDClassifier` with `joblib` between epochs.
3. The API stays unchanged, since the artifact is still `FeatureBuilder -> classifier` and
   `FeatureBuilder` no longer carries a vocabulary of millions of entries.

## Honest status

This experiment has **not** been run against the Kaggle data in this repository, so there is no
measured score for it. The script was only smoke-tested on a tiny synthetic CSV. Expect it to land
somewhat below the 0.826 TF-IDF stack (no IDF, single linear model); the point is the memory profile,
not the score. Running it on `data/train.csv` and recording the held-out Macro F1 here would complete
the comparison.
