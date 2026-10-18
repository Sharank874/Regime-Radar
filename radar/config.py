"""Central configuration. Every modelling choice lives here so it is fixed
*before* evaluation (no tuning on the test period)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RADAR_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = ROOT / "reports"
WEB_DIST = ROOT / "web" / "dist"

for _d in (DATA_DIR, CACHE_DIR, RAW_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Yahoo Finance tickers
TICKERS = {
    "nifty": "^NSEI",
    "india_vix": "^INDIAVIX",
    "spx": "^GSPC",
    "us_vix": "^VIX",
}
HISTORY_START = "2007-09-17"  # first NIFTY 50 bar on Yahoo
DEMO_SEED = 4                 # simulator seed for demo mode (any seed works; this one ends in a fresh shock)


@dataclass(frozen=True)
class ModelConfig:
    # --- target -------------------------------------------------------
    horizon: int = 5                 # trading days ahead
    stress_quantile: float = 0.80    # "high-vol week" = top 20% of forward RV in the TRAINING window
    # --- HMM ----------------------------------------------------------
    n_states: int = 3
    hmm_features: tuple = ("ret", "log_rv5", "log_vix", "vix_chg5")  # index 1 (vol) orders states
    hmm_news_feature: str = "news_stress"
    hmm_inits: int = 6
    hmm_iter: int = 150
    hmm_sticky: float = 50.0         # Dirichlet pseudo-counts on the diagonal -> persistent regimes
    # --- classifier ---------------------------------------------------
    clf_features: tuple = (
        "p_stress", "p_calm", "log_rv5", "log_rv22", "rv_ratio", "dd252",
        "mom20", "log_vix", "vix_chg5", "vrp", "spx_ret_lag", "spx_absret_lag",
    )
    clf_news_features: tuple = ("news_stress", "news_tone", "news_surprise_max")
    clf_C: float = 0.5
    n_bootstrap: int = 25            # bootstrap models for the uncertainty band
    block_len: int = 20              # block bootstrap block length (days)
    # --- walk-forward -------------------------------------------------
    min_train_years: int = 5
    first_test_year: int = 2014
    news_first_year: int = 2017      # GDELT timelines start 2017-01-01
    seed: int = 7


CFG = ModelConfig()

STATE_NAMES = ["Calm", "Choppy", "Stress"]

FEATURE_LABELS = {
    "ret": "Daily return",
    "log_rv5": "Short-term volatility",
    "log_rv22": "Monthly volatility",
    "rv_ratio": "Vol acceleration",
    "dd252": "Drawdown from 1y high",
    "mom20": "20-day momentum",
    "ma50_dist": "Distance from 50-day avg",
    "log_vix": "India VIX level",
    "vix_chg5": "India VIX 5-day change",
    "vrp": "VIX minus realized vol",
    "spx_ret_lag": "S&P 500 last session",
    "spx_absret_lag": "S&P 500 move size",
    "p_stress": "HMM stress probability",
    "p_calm": "HMM calm probability",
    "news_stress": "News stress",
    "news_tone": "News tone",
    "news_surprise_max": "Biggest news surprise",
}
