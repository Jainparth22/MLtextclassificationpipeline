"""
Serving-ready version of the submitted V5 feature pipeline.

Reproduces the feature logic of src/milestone_5_final.py as one sklearn transformer, so
that raw records go in and a single joblib artifact can be trained, saved and served:

    raw DataFrame -> FeatureBuilder -> StackingClassifier

Features (60,019 dims with the default sizes):
    35,000 word TF-IDF (1-3 grams) + 25,000 char_wb TF-IDF (3-5 grams)
    + 14 numeric (log1p on skewed ones, then StandardScaler) + 5 binary.

This module must be importable at load time (joblib/pickle stores the class by reference),
so both train_export.py and app.py import it from the same directory.
"""
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

LABEL_NAMES = {
    0: "neutral",
    1: "mildly_problematic",
    2: "moderately_toxic",
    3: "severely_toxic",
}

# Raw input columns the pipeline understands. Only `comment` is required; the rest
# default to the values below (the "missing" encoding used in training).
RAW_DEFAULTS = {
    "comment": "",
    "created_date": None,
    "upvote": 0,
    "downvote": 0,
    "if_1": 0,
    "if_2": 0,
    "emoticon_1": 0,
    "emoticon_2": 0,
    "emoticon_3": 0,
    "disability": 0,
    "race": None,
    "religion": None,
    "gender": None,
}

NUMERIC_FEATURES = [
    "upvote", "downvote", "if_1", "if_2",
    "comment_length", "word_count", "avg_word_length",
    "total_votes", "vote_diff", "vote_ratio",
    "total_emoticons", "hour", "day_of_week", "month",
]
BINARY_FEATURES = ["emoticon_1", "emoticon_2", "emoticon_3", "disability", "has_detection"]
SKEWED_COLS = ["upvote", "downvote", "if_1", "if_2",
               "total_votes", "vote_ratio", "comment_length", "word_count"]


def clean_text(text):
    """Minimal cleaning used by the submitted model: no stemming, no lemmatization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text_keep_punct(text):
    """Variant that keeps '*', '!' and '?' so char n-grams can see masked profanity (f**k) and emphasis.
    The submitted V5 model uses clean_text (which turns these into spaces); this variant exists
    to test that design choice (see src/evaluate.py --step ablation)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s'*!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


CLEANERS = {"v5": clean_text, "keep_punct": clean_text_keep_punct}


def _structured(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the 14 numeric + 5 binary structured features from raw columns."""
    df = df.copy()
    for col, default in RAW_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df["comment"] = df["comment"].fillna("").astype(str)

    dt = pd.to_datetime(df["created_date"], errors="coerce")
    df["hour"] = dt.dt.hour.fillna(0)
    df["day_of_week"] = dt.dt.dayofweek.fillna(0)
    df["month"] = dt.dt.month.fillna(0)

    df["comment_length"] = df["comment"].str.len()
    df["word_count"] = df["comment"].str.split().str.len().fillna(0).astype(int)
    df["avg_word_length"] = df["comment_length"] / (df["word_count"] + 1)
    df["total_votes"] = df["upvote"] + df["downvote"]
    df["vote_diff"] = df["upvote"] - df["downvote"]
    df["vote_ratio"] = df["upvote"] / (df["downvote"] + 1)
    df["total_emoticons"] = df["emoticon_1"] + df["emoticon_2"] + df["emoticon_3"]

    demo = df[["race", "religion", "gender"]].fillna("missing")
    df["has_detection"] = demo.ne("missing").any(axis=1).astype(int)
    df["disability"] = df["disability"].fillna(0).astype(int)
    return df


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw records (DataFrame) -> sparse matrix identical to the V5 training features.

    The use_* flags and clean_mode exist for ablation studies; the defaults are the submitted config.
    """

    def __init__(self, word_features=35000, char_features=25000, min_df=3,
                 use_word=True, use_char=True, use_structured=True, clean_mode="v5"):
        self.word_features = word_features
        self.char_features = char_features
        self.min_df = min_df
        self.use_word = use_word
        self.use_char = use_char
        self.use_structured = use_structured
        self.clean_mode = clean_mode

    def _text_and_dense(self, df, fit):
        s = _structured(df)
        text = s["comment"].map(CLEANERS[self.clean_mode])
        dense = []
        if self.use_structured:
            num = s[NUMERIC_FEATURES].to_numpy(dtype=float).copy()
            for col in SKEWED_COLS:
                i = NUMERIC_FEATURES.index(col)
                num[:, i] = np.log1p(np.clip(num[:, i], 0, None))
            num = self.scaler_.fit_transform(num) if fit else self.scaler_.transform(num)
            dense = [csr_matrix(num), csr_matrix(s[BINARY_FEATURES].to_numpy(dtype=float))]
        return text, dense

    def fit_transform(self, X, y=None):
        if not (self.use_word or self.use_char or self.use_structured):
            raise ValueError("at least one feature block must be enabled")
        self.word_ = TfidfVectorizer(
            max_features=self.word_features, ngram_range=(1, 3),
            min_df=self.min_df, max_df=0.9, sublinear_tf=True)
        self.char_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=self.char_features,
            min_df=self.min_df, max_df=0.9, sublinear_tf=True)
        self.scaler_ = StandardScaler()
        text, dense = self._text_and_dense(X, fit=True)
        blocks = []
        if self.use_word:
            blocks.append(self.word_.fit_transform(text))
        if self.use_char:
            blocks.append(self.char_.fit_transform(text))
        return hstack(blocks + dense).tocsr()

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        text, dense = self._text_and_dense(X, fit=False)
        blocks = []
        if self.use_word:
            blocks.append(self.word_.transform(text))
        if self.use_char:
            blocks.append(self.char_.transform(text))
        return hstack(blocks + dense).tocsr()
