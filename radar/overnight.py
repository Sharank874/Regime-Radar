"""Overnight bridge: the US session closes in the early hours (IST), before NIFTY opens.
Given the latest S&P 500 session, what has NIFTY's next session historically looked like?"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = [(-np.inf, -0.02, "S&P fell more than 2%"), (-0.02, -0.01, "S&P fell 1–2%"),
           (-0.01, -0.003, "S&P fell 0.3–1%"), (-0.003, 0.003, "S&P roughly flat"),
           (0.003, 0.01, "S&P rose 0.3–1%"), (0.01, 0.02, "S&P rose 1–2%"), (0.02, np.inf, "S&P rose more than 2%")]


def _bucket(x: float) -> int:
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo < x <= hi:
            return i
    return 3


def overnight(df: pd.DataFrame, spx: pd.DataFrame, us_vix: pd.DataFrame, asof: pd.Timestamp | None = None) -> dict:
    hist = df[["ret", "spx_ret_lag", "close", "open"]].dropna(subset=["ret", "spx_ret_lag"]).copy()
    hist["gap"] = hist["open"] / hist["close"].shift(1) - 1
    hist = hist.iloc[1:]
    gap_ok = (hist["gap"].abs() > 1e-6).mean() > 0.8          # Yahoo opens for ^NSEI are sometimes = prev close
    hist["b"] = hist["spx_ret_lag"].map(_bucket)
    table = []
    for i, (lo, hi, label) in enumerate(BUCKETS):
        g = hist[hist["b"] == i]
        if len(g) < 10:
            table.append({"bucket": label, "n": int(len(g))})
            continue
        table.append({"bucket": label, "n": int(len(g)),
                      "p_big_move": float((g["ret"].abs() > 0.01).mean()),
                      "p_down": float((g["ret"] < 0).mean()),
                      "median_ret": float(g["ret"].median()),
                      "median_gap": float(g["gap"].median()) if gap_ok else None})
    base = {"p_big_move": float((hist["ret"].abs() > 0.01).mean()), "p_down": float((hist["ret"] < 0).mean())}

    asof = asof or df.index[-1]
    s = np.log(spx["close"]).diff()
    after = s[s.index >= asof]                 # US session on the same calendar date closes after NIFTY
    sess_date = after.index[-1] if len(after) else s.index[s.index < asof][-1]
    r = float(s.loc[sess_date])
    v = us_vix["close"]
    vchg = float(np.log(v.loc[:sess_date].iloc[-1] / v.loc[:sess_date].iloc[-2])) if len(v.loc[:sess_date]) > 1 else None
    b = _bucket(r)
    pct = float((hist["spx_ret_lag"] <= r).mean())
    abs_pct = float((hist["spx_ret_lag"].abs() < abs(r)).mean())
    return {"spx_session": str(sess_date.date()), "spx_ret": r, "spx_pctile": pct, "spx_abs_pctile": abs_pct, "us_vix_chg": vchg,
            "bucket": b, "bucket_label": BUCKETS[b][2], "conditional": table[b], "base": base,
            "table": table, "gap_reliable": bool(gap_ok), "nifty_last_close": str(asof.date())}
