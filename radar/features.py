"""Leakage-safe feature engineering.

Rule: the feature row for NIFTY date t uses only information available at the
NIFTY close (15:30 IST) on date t.
  * rolling windows end at t; drawdown uses a trailing 252-day peak (not the full-sample peak)
  * S&P 500 values come from the latest US session that closed *before* date t
  * news for date t comes from UTC days strictly before date t
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import CATEGORIES

ANN = np.sqrt(252)


def _align_prior(target_index: pd.DatetimeIndex, s: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """For each target date, take the latest row of `s` with date < target date."""
    left = pd.DataFrame({"date": target_index})
    right = s.to_frame() if isinstance(s, pd.Series) else s.copy()
    right = right.rename_axis("src_date").reset_index()
    right["date"] = right["src_date"]
    out = pd.merge_asof(left.sort_values("date"), right.sort_values("date"), on="date",
                        allow_exact_matches=False, direction="backward")
    return out.set_index("date")


def news_features(news: pd.DataFrame) -> pd.DataFrame:
    """Per-category surprise (z-score of log volume share vs trailing 60 days) and tone."""
    out = pd.DataFrame(index=news.index)
    total = news["vol_total"].clip(lower=1)
    surprises, tones = [], []
    for c in CATEGORIES:
        vcol, tcol = f"vol_{c.key}", f"tone_{c.key}"
        if vcol not in news:
            continue
        share = np.log((news[vcol] + 1) / total)
        mu = share.rolling(60, min_periods=20).mean().shift(1)
        sd = share.rolling(60, min_periods=20).std().shift(1)
        z = ((share - mu) / sd.replace(0, np.nan)).clip(-4, 4)
        out[f"surprise_{c.key}"] = z
        surprises.append(z)
        if tcol in news:
            tz_mu = news[tcol].rolling(60, min_periods=20).mean().shift(1)
            tz_sd = news[tcol].rolling(60, min_periods=20).std().shift(1)
            tz = ((news[tcol] - tz_mu) / tz_sd.replace(0, np.nan)).clip(-4, 4)
            out[f"tonez_{c.key}"] = tz
            tones.append(tz)
    S = pd.concat(surprises, axis=1)
    T = pd.concat(tones, axis=1) if tones else S * 0
    out["news_tone"] = T.mean(axis=1)
    out["news_surprise_max"] = S.max(axis=1)
    # stress = how unusual the coverage is, plus how negative it is (both standardised)
    out["news_stress"] = 0.5 * S.clip(lower=0).mean(axis=1) * 2 - 0.5 * out["news_tone"]
    return out


def build_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nifty = data["nifty"]
    df = pd.DataFrame(index=nifty.index)
    c = nifty["close"]
    df["close"] = c
    df["open"] = nifty.get("open", c)
    ret = np.log(c).diff()
    df["ret"] = ret
    df["rv5"] = ret.rolling(5).std() * ANN
    df["rv22"] = ret.rolling(22).std() * ANN
    df["log_rv5"] = np.log(df["rv5"].clip(lower=1e-4))
    df["log_rv22"] = np.log(df["rv22"].clip(lower=1e-4))
    df["rv_ratio"] = np.log(df["rv5"] / df["rv22"])
    peak = c.rolling(252, min_periods=20).max()
    df["dd252"] = c / peak - 1
    df["mom20"] = np.log(c / c.shift(20))
    df["ma50_dist"] = c / c.rolling(50).mean() - 1
    downside = ret.where(ret < 0, 0.0)
    df["dvol22"] = np.sqrt((downside ** 2).rolling(22).mean()) * ANN

    # India VIX (same-day close is known at NIFTY close)
    vix = data["india_vix"]["close"].reindex(df.index).ffill(limit=3)
    df["vix"] = vix
    df["log_vix"] = np.log(vix)
    df["vix_chg5"] = np.log(vix / vix.shift(5))
    df["vrp"] = vix / 100 - df["rv22"]

    # S&P 500: last completed US session before this NIFTY date
    spx = data["spx"]["close"]
    spx_ret = np.log(spx).diff().rename("spx_ret")
    spx_feat = pd.DataFrame({"spx_ret": spx_ret,
                             "spx_rv22": spx_ret.rolling(22).std() * ANN})
    al = _align_prior(df.index, spx_feat)
    df["spx_ret_lag"] = al["spx_ret"].to_numpy()
    df["spx_absret_lag"] = np.abs(df["spx_ret_lag"])
    df["spx_rv22_lag"] = al["spx_rv22"].to_numpy()

    # news (optional; available 2017+)
    if "news_daily" in data and data["news_daily"] is not None and len(data["news_daily"]):
        nf = news_features(data["news_daily"])
        al = _align_prior(df.index, nf).drop(columns=["src_date"])
        for col in al.columns:
            df[col] = al[col].to_numpy()
    else:
        for col in ("news_stress", "news_tone", "news_surprise_max"):
            df[col] = np.nan

    return df


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward realised vol over t+1..t+h (annualised). The *threshold* is set per
    training fold, so only the continuous forward RV is stored here."""
    r = df["ret"]
    fwd = pd.concat([r.shift(-k) for k in range(1, horizon + 1)], axis=1)
    df = df.copy()
    df["fwd_rv"] = fwd.std(axis=1, ddof=1) * ANN
    df.loc[fwd.isna().any(axis=1), "fwd_rv"] = np.nan
    df["fwd_absret1"] = r.shift(-1).abs()
    return df
