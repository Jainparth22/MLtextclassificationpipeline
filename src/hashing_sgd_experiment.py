"""
Scalability experiment: stateless HashingVectorizer + out-of-core SGDClassifier.

Why: TfidfVectorizer keeps a vocabulary dict in RAM and needs the whole corpus to fit, so it
does not scale to 10M+ rows. HashingVectorizer has no vocabulary (token -> hash -> column),
so memory is fixed by `n_features` and the corpus can be streamed in chunks.

    python src/hashing_sgd_experiment.py --train data/train.csv --chunksize 20000

Trade-offs vs the submitted TF-IDF model (this script was NOT used for the 0.826 score):
  * no IDF weighting (IDF needs a global pass; could be approximated with a first pass of
    document-frequency counts or dropped) -> expect a small accuracy loss
  * hash collisions at n_features=2**20 (tunable; 2**22+ for bigger corpora)
  * partial_fit does not support class_weight='balanced', so balanced weights are passed
    per-sample (computed from one cheap pass over the label column)
  * no calibrated/stacked ensemble; a single linear model
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from toxic_pipeline import clean_text

CLASSES = np.array([0, 1, 2, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--n-features", type=int, default=2 ** 20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--val-every", type=int, default=5, help="every k-th chunk is held out")
    args = ap.parse_args()

    # pass 1: label counts only (cheap) -> balanced class weights
    counts = pd.read_csv(args.train, usecols=["label"])["label"].value_counts().reindex(CLASSES)
    weights = counts.sum() / (len(CLASSES) * counts)
    print("balanced weights:", weights.round(2).to_dict())

    hv = HashingVectorizer(n_features=args.n_features, ngram_range=(1, 2),
                           alternate_sign=False, norm="l2", preprocessor=None)
    clf = SGDClassifier(loss="modified_huber", alpha=1e-5, random_state=42)

    def chunks():
        for i, ch in enumerate(pd.read_csv(args.train, chunksize=args.chunksize)):
            ch = ch.dropna(subset=["comment"])
            yield i, hv.transform(ch["comment"].map(clean_text)), ch["label"].to_numpy()

    for epoch in range(args.epochs):
        for i, X, y in chunks():
            if i % args.val_every == 0:
                continue  # held out
            clf.partial_fit(X, y, classes=CLASSES, sample_weight=weights.loc[y].to_numpy())
        print(f"epoch {epoch + 1} done")

    ys, ps = [], []
    for i, X, y in chunks():
        if i % args.val_every == 0:
            ys.append(y)
            ps.append(clf.predict(X))
    print("held-out Macro F1:", round(f1_score(np.concatenate(ys), np.concatenate(ps),
                                               average="macro"), 4))


if __name__ == "__main__":
    main()
