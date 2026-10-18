"""Realistic synthetic market used for offline development, tests and the demo mode.

It is NOT market data. It reproduces the *stylised facts* the models rely on
(regime switching, volatility clustering, fat tails, an implied-vol index that
partly anticipates realised vol, overnight US->India transmission, and news
volume that rises around stress) so the full pipeline can be exercised and
validated end-to-end without network access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import CATEGORIES

# daily drift, daily vol per regime: calm / choppy / stress
_MU = np.array([0.0006, 0.0000, -0.0022])
_SIG = np.array([0.0072, 0.0125, 0.0255])
_A = np.array([
    [0.986, 0.012, 0.002],
    [0.030, 0.955, 0.015],
    [0.010, 0.060, 0.930],
])
# news categories that tend to *lead* stress (a weak, realistic signal)
_LEADERS = {"geopolitics": 1.6, "flows": 1.2, "global_rates": 1.0, "market_stress": 2.2, "monetary": 0.6}


def _markov_chain(n: int, rng: np.random.Generator) -> np.ndarray:
    s = np.empty(n, dtype=int)
    s[0] = 0
    cum = _A.cumsum(1)
    u = rng.random(n)
    for t in range(1, n):
        s[t] = int(np.searchsorted(cum[s[t - 1]], u[t]))
    return s


def simulate_market(start="2007-09-17", end=None, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.today().normalize() - pd.offsets.BDay(1)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # --- US market: its own chain, NIFTY regime partially follows it ---------
    s_us = _markov_chain(n, rng)
    s_in = s_us.copy()
    flip = rng.random(n) < 0.22                       # idiosyncratic Indian regime days
    s_in[flip] = _markov_chain(n, rng)[flip]
    # smooth idiosyncratic flips into spells
    s_in = pd.Series(s_in).rolling(5, min_periods=1).median().round().astype(int).to_numpy()

    # within-regime vol clustering (log-AR(1)) and student-t shocks
    h = np.zeros(n)
    eps_h = rng.normal(0, 0.12, n)
    for t in range(1, n):
        h[t] = 0.94 * h[t - 1] + eps_h[t]
    t5 = rng.standard_t(5, n) / np.sqrt(5 / 3)
    t5b = rng.standard_t(5, n) / np.sqrt(5 / 3)

    spx_ret = _MU[s_us] * 0.9 + _SIG[s_us] * 0.95 * np.exp(h * 0.8) * t5b
    sig_in = _SIG[s_in] * np.exp(h)
    nifty_ret = _MU[s_in] + 0.33 * np.r_[0, spx_ret[:-1]] + sig_in * t5 * 0.94

    nifty = 4500 * np.exp(np.cumsum(nifty_ret))
    spx = 1500 * np.exp(np.cumsum(spx_ret))

    # OHLC around the close
    def ohlc(close, ret, sig, gap_from=None):
        prev = np.r_[close[0], close[:-1]]
        gap = (gap_from if gap_from is not None else 0) + rng.normal(0, 0.25, len(close)) * sig
        open_ = prev * np.exp(gap)
        rng_ = np.abs(rng.normal(0, 1, len(close))) * sig * 0.8
        high = np.maximum(open_, close) * np.exp(rng_)
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 1, len(close))) * sig * 0.8)
        return open_, high, low

    o, hi, lo = ohlc(nifty, nifty_ret, sig_in, gap_from=0.6 * np.r_[0, spx_ret[:-1]])
    vol = (2.5e5 * (1 + 0.8 * (s_in == 2)) * np.exp(rng.normal(0, 0.2, n))).astype(int)
    nifty_df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": nifty, "volume": vol}, index=dates)

    so, shi, slo = ohlc(spx, spx_ret, _SIG[s_us])
    spx_df = pd.DataFrame({"open": so, "high": shi, "low": slo, "close": spx, "volume": 0}, index=dates)

    # --- implied vol: market's expectation of next-5d vol + risk premium + noise ----
    P5 = np.linalg.matrix_power(_A, 5)
    exp_sig = (P5[s_in] @ _SIG) * np.exp(h)
    india_vix = np.sqrt(252) * 100 * exp_sig * 1.12 * np.exp(rng.normal(0, 0.07, n))
    india_vix = pd.Series(india_vix).ewm(span=3).mean().to_numpy()
    us_exp = (P5[s_us] @ _SIG) * 0.95 * np.exp(h * 0.8)
    us_vix = np.sqrt(252) * 100 * us_exp * 1.15 * np.exp(rng.normal(0, 0.07, n))
    vix_df = pd.DataFrame({"close": india_vix}, index=dates)
    usvix_df = pd.DataFrame({"close": us_vix}, index=dates)

    # --- news: daily volume & tone per category (2017+ like GDELT) -----------------
    stress_next = pd.Series((s_in == 2).astype(float)).shift(-3).rolling(4, min_periods=1).mean().fillna(0).to_numpy()
    stress_now = (s_in == 2).astype(float)
    rows = {}
    for c in CATEGORIES:
        lead = _LEADERS.get(c.key, 0.2)
        lam = 40 * np.exp(rng.normal(0, 0.15, n)) * (1 + lead * (0.6 * stress_next + 0.6 * stress_now))
        rows[f"vol_{c.key}"] = rng.poisson(lam)
        rows[f"tone_{c.key}"] = rng.normal(-0.6 - 1.4 * (0.5 * stress_next + 0.5 * stress_now) * min(lead, 1.5), 1.1, n)
    news = pd.DataFrame(rows, index=dates)
    news["vol_total"] = 4000 * np.exp(rng.normal(0, 0.1, n))
    news = news.loc[news.index >= "2017-01-01"]

    regimes = pd.DataFrame({"true_regime": s_in}, index=dates)
    return {"nifty": nifty_df, "india_vix": vix_df, "spx": spx_df, "us_vix": usvix_df,
            "news_daily": news, "true_regime": regimes}
