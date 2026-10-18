"""Append-only prediction ledger.

Every live forecast is written once, with a timestamp, BEFORE its outcome is known,
and scored automatically when the 5-session horizon has passed. Back-tested
(walk-forward) predictions are kept in the same table but tagged `backtest`, and the
UI never mixes the two.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import DATA_DIR


def _con(mode: str) -> sqlite3.Connection:
    con = sqlite3.connect(DATA_DIR / f"ledger_{mode}.sqlite")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        asof TEXT, kind TEXT, created_utc TEXT, p REAL, p_lo REAL, p_hi REAL, threshold REAL,
        model_version TEXT, regime TEXT, outcome REAL, realized_vol REAL, scored_utc TEXT,
        PRIMARY KEY(asof, kind))""")
    return con


def record_live(mode: str, asof: pd.Timestamp, row: pd.Series, regime: str, version: str) -> bool:
    """Write today's forecast if not already present. Returns True if inserted."""
    con = _con(mode)
    cur = con.execute("INSERT OR IGNORE INTO predictions(asof,kind,created_utc,p,p_lo,p_hi,threshold,model_version,regime)"
                      " VALUES (?,?,?,?,?,?,?,?,?)",
                      (str(asof.date()), "live", datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       float(row["p_model"]), float(row["p_lo"]), float(row["p_hi"]), float(row["threshold"]),
                       version, regime))
    con.commit()
    ok = cur.rowcount > 0
    con.close()
    return ok


def sync_backtest(mode: str, tl: pd.DataFrame, version: str, regimes: pd.Series):
    con = _con(mode)
    con.execute("DELETE FROM predictions WHERE kind='backtest'")
    rows = [(str(d.date()), "backtest", None, float(r.p_model), float(r.p_lo), float(r.p_hi), float(r.threshold),
             version, regimes.get(d, ""), None if np.isnan(r.y) else float(r.y), None, None)
            for d, r in tl.iterrows() if not np.isnan(r.p_model)]
    con.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def score_live(mode: str, df: pd.DataFrame):
    """Fill outcomes for live predictions whose horizon has fully elapsed."""
    con = _con(mode)
    pend = con.execute("SELECT asof, threshold FROM predictions WHERE kind='live' AND outcome IS NULL").fetchall()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for asof, thr in pend:
        d = pd.Timestamp(asof)
        if d in df.index and not np.isnan(df.loc[d, "fwd_rv"]):
            rv = float(df.loc[d, "fwd_rv"])
            con.execute("UPDATE predictions SET outcome=?, realized_vol=?, scored_utc=? WHERE asof=? AND kind='live'",
                        (float(rv > thr), rv, now, asof))
    con.commit()
    con.close()


def read(mode: str, kind: str) -> pd.DataFrame:
    con = _con(mode)
    df = pd.read_sql_query("SELECT * FROM predictions WHERE kind=? ORDER BY asof", con, params=(kind,))
    con.close()
    return df
