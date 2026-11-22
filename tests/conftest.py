import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def raw_df():
    rng = np.random.default_rng(0)
    words = {0: "nice great thanks good article", 1: "meh silly wrong odd",
             2: "stupid dumb hate bad", 3: "kill idiot scum die trash"}
    filler = "the a is you this".split()
    rows = []
    for i in range(400):
        label = int(rng.choice(4, p=[.4, .2, .2, .2]))
        text = " ".join(rng.choice(words[label].split() + filler, size=10))
        rows.append(dict(comment=text, created_date=f"2020-01-{i % 9 + 1:02d} 10:00:00",
                         upvote=i % 5, downvote=i % 3, if_1=i % 7, if_2=i % 4,
                         emoticon_1=0, emoticon_2=i % 2, emoticon_3=0, disability=False,
                         race=None, religion=None, gender="male" if i % 4 == 0 else None, label=label))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_model(raw_df, tmp_path_factory):
    """Small, fast pipeline with the real FeatureBuilder (LogReg instead of the slow stack)."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from toxic_pipeline import FeatureBuilder
    X, y = raw_df.drop(columns="label"), raw_df["label"]
    model = Pipeline([("features", FeatureBuilder(500, 500, 2)),
                      ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    path = tmp_path_factory.mktemp("model") / "final_stack_pipeline.joblib"
    joblib.dump(model, path)
    return str(path)
