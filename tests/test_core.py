"""Core guarantees: no look-ahead, exact attributions, sane fold boundaries."""
import numpy as np
import pandas as pd
import pytest

from radar.features import add_target, build_features
from radar.hmm import GaussianHMM, shapley_update
from radar.simulate import simulate_market


@pytest.fixture(scope="module")
def data():
    return simulate_market(start="2012-01-02", end="2019-12-31", seed=3)


def test_features_have_no_lookahead(data):
    """Changing prices AFTER a cutoff must not change any feature on or before the cutoff."""
    base = build_features(data)
    cut = base.index[1200]
    shocked = {k: v.copy() for k, v in data.items()}
    for k in ("nifty", "india_vix", "spx", "us_vix"):
        m = shocked[k].index > cut
        shocked[k].loc[m, "close"] = shocked[k].loc[m, "close"] * 1.37
    news_m = shocked["news_daily"].index >= cut
    shocked["news_daily"].loc[news_m] = shocked["news_daily"].loc[news_m] * 3
    alt = build_features(shocked)
    a, b = base.loc[:cut], alt.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)


def test_target_uses_only_future(data):
    df = add_target(build_features(data), 5)
    t = df.index[500]
    expected = np.std(df["ret"].iloc[501:506], ddof=1) * np.sqrt(252)
    assert np.isclose(df.loc[t, "fwd_rv"], expected)


def test_hmm_filter_is_causal_and_shapley_exact(data):
    df = build_features(data)
    X = df[["ret", "log_rv5", "log_vix", "vix_chg5"]].dropna()
    Z = ((X - X.mean()) / X.std()).to_numpy()
    m = GaussianHMM(n_init=2, seed=0).fit(Z[:1000])
    prior, post, L = m.filter(Z)
    prior2, post2, _ = m.filter(Z[:800])
    assert np.allclose(post[:800], post2)                  # future rows don't change past beliefs
    phi = shapley_update(prior, L)
    assert np.allclose(prior + phi.sum(-1), post, atol=1e-10)


def test_hmm_handles_missing_features(data):
    df = build_features(data)
    X = df[["ret", "log_rv5", "log_vix", "news_stress"]].dropna(subset=["ret", "log_rv5", "log_vix"])
    Z = ((X - X.mean()) / X.std()).to_numpy()
    assert np.isnan(Z[:, 3]).any()
    m = GaussianHMM(n_init=2, seed=0).fit(Z)
    _, post, L = m.filter(Z)
    assert np.isfinite(post).all()
    assert (L[np.isnan(Z[:, 3]), :, 3] == 0).all()


def test_fold_boundaries(data):
    from radar.config import CFG
    from radar.pipeline import _fit_fold
    df = add_target(build_features(data), 5)
    f = _fit_fold(df, 2018, CFG, use_news=False)
    assert f.train_end < pd.Timestamp("2018-01-01")
    # purge: the last training target window ends before the test year starts
    last_target_day = df.index[df.index.get_loc(f.train_end) + 5]
    assert last_target_day < pd.Timestamp("2018-01-01")
    tr = df.loc[:f.train_end, "fwd_rv"].dropna()
    assert np.isclose(f.threshold, tr.loc[tr.index.isin(df.dropna(subset=list(CFG.hmm_features)).index)].quantile(0.8))
