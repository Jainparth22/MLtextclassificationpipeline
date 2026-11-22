import pytest

from toxic_pipeline import (BINARY_FEATURES, NUMERIC_FEATURES, FeatureBuilder, clean_text,
                            clean_text_keep_punct)


def test_clean_text_v5_drops_noise_and_symbols():
    assert clean_text("Visit http://x.com NOW <b>ok</b> a@b.com 123") == "visit now ok"
    assert clean_text(None) == ""
    # documented behaviour: V5 cleaning turns '*' into a space (masked profanity is split)
    assert clean_text("f**k") == "f k"


def test_keep_punct_variant_preserves_masking_and_emphasis():
    assert clean_text_keep_punct("F**k YOU!!") == "f**k you!!"


def test_feature_dims_and_train_test_consistency(raw_df):
    X = raw_df.drop(columns="label")
    fb = FeatureBuilder(300, 200, 2).fit(X)
    A = fb.transform(X)
    assert A.shape[1] == len(fb.word_.vocabulary_) + len(fb.char_.vocabulary_) \
        + len(NUMERIC_FEATURES) + len(BINARY_FEATURES)
    assert fb.transform(X.iloc[:3]).shape == (3, A.shape[1])


def test_transform_does_not_refit(raw_df):
    X = raw_df.drop(columns="label")
    fb = FeatureBuilder(300, 200, 2).fit(X.iloc[:200])
    vocab = dict(fb.word_.vocabulary_)
    fb.transform(X.iloc[200:])
    assert fb.word_.vocabulary_ == vocab


def test_only_comment_required(raw_df):
    X = raw_df[["comment"]]
    fb = FeatureBuilder(300, 200, 2).fit(raw_df.drop(columns="label"))
    assert fb.transform(X.iloc[:2]).shape[0] == 2


@pytest.mark.parametrize("kw,expect_blocks", [({"use_word": False}, 2), ({"use_char": False}, 2),
                                              ({"use_structured": False}, 2)])
def test_ablation_flags_change_dimension(raw_df, kw, expect_blocks):
    X = raw_df.drop(columns="label")
    full = FeatureBuilder(300, 200, 2).fit_transform(X).shape[1]
    assert FeatureBuilder(300, 200, 2, **kw).fit_transform(X).shape[1] < full


def test_no_blocks_is_an_error(raw_df):
    with pytest.raises(ValueError):
        FeatureBuilder(use_word=False, use_char=False, use_structured=False).fit(
            raw_df.drop(columns="label"))
