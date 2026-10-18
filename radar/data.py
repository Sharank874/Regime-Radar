"""Market data loading: Yahoo Finance (live mode) or the simulator (demo mode).

Raw downloads are cached as CSV in data/raw/ with the download time recorded in
data/raw/provenance.json so every report can state its data source and date.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from .config import HISTORY_START, RAW_DIR, TICKERS

log = logging.getLogger("radar.data")
PROVENANCE = RAW_DIR / "provenance.json"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    return df


def fetch_yahoo(refresh: bool = True) -> dict[str, pd.DataFrame]:
    """Download (or reuse cached) daily bars for every ticker."""
    import yfinance as yf  # imported lazily: not needed in demo mode

    out, prov = {}, {}
    for name, tick in TICKERS.items():
        path = RAW_DIR / f"{name}.csv"
        cached = None
        if path.exists():
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if refresh or cached is None:
            try:
                df = yf.download(tick, start=HISTORY_START, auto_adjust=False, progress=False, threads=False)
                df = _clean(df)
                if len(df) > 100:
                    df.to_csv(path)
                    cached = df
                    prov[name] = {"ticker": tick, "rows": len(df), "first": str(df.index[0].date()),
                                  "last": str(df.index[-1].date()),
                                  "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            except Exception as e:  # network errors etc: fall back to cache
                log.warning("download failed for %s: %s", tick, e)
        if cached is None:
            raise RuntimeError(f"No data for {tick}. Check your internet connection and try `radar fetch`.")
        out[name] = _clean(cached)
    if prov:
        old = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
        old.update(prov)
        PROVENANCE.write_text(json.dumps(old, indent=2))
    return out


def fetch_intraday_last() -> dict[str, dict]:
    """Latest (delayed) intraday print for NIFTY and India VIX -> provisional nowcast."""
    import yfinance as yf

    res = {}
    for name in ("nifty", "india_vix"):
        try:
            df = yf.download(TICKERS[name], period="2d", interval="5m", progress=False, threads=False)
            df = _clean_intraday(df)
            if len(df):
                res[name] = {"time": df.index[-1].isoformat(), "price": float(df["close"].iloc[-1])}
        except Exception as e:
            log.warning("intraday fetch failed for %s: %s", name, e)
    return res


def _clean_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df[df["close"].notna()]


def load_market(mode: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """mode = 'live' (Yahoo) or 'demo' (simulated)."""
    if mode == "demo":
        from .config import DEMO_SEED
        from .simulate import simulate_market
        return simulate_market(seed=DEMO_SEED)
    data = fetch_yahoo(refresh=refresh)
    news_path = RAW_DIR / "news_daily.csv"
    if news_path.exists():
        data["news_daily"] = pd.read_csv(news_path, index_col=0, parse_dates=True)
    return data


def provenance(mode: str) -> dict:
    if mode == "demo":
        return {"source": "SIMULATED market (radar/simulate.py) — not real data"}
    p = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
    return {"source": "Yahoo Finance via yfinance; news via GDELT DOC 2.0 + RSS", "series": p}
