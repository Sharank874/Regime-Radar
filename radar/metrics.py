"""Evaluation metrics for probabilistic stress forecasts, with block-bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .pipeline import block_bootstrap_idx


def _clean(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    return y[ok].astype(int), np.clip(p[ok], 1e-6, 1 - 1e-6)


def score(y, p, p_clim=None) -> dict:
    y, p = _clean(y, p)
    if len(y) < 30 or y.min() == y.max():
        return {"n": int(len(y))}
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }
    ref = np.full_like(p, y.mean()) if p_clim is None else np.clip(np.asarray(p_clim, float)[: len(p)], 1e-6, 1)
    out["brier_skill"] = 1 - out["brier"] / brier_score_loss(y, np.full_like(p, y.mean()))
    return out


def paired_diff_ci(y, p_a, p_b, metric: str = "brier", block: int = 20, n_boot: int = 500, seed: int = 0) -> dict:
    """Block-bootstrap CI for metric(a) - metric(b). For brier, negative = a better."""
    y = np.asarray(y, float)
    pa, pb = np.asarray(p_a, float), np.asarray(p_b, float)
    ok = ~(np.isnan(y) | np.isnan(pa) | np.isnan(pb))
    y, pa, pb = y[ok].astype(int), pa[ok], pb[ok]
    f = {"brier": lambda yy, pp: brier_score_loss(yy, pp),
         "pr_auc": lambda yy, pp: average_precision_score(yy, pp)}[metric]
    rng = np.random.default_rng(seed)
    point = f(y, pa) - f(y, pb)
    ds = []
    for _ in range(n_boot):
        idx = block_bootstrap_idx(len(y), block, rng)
        if y[idx].min() == y[idx].max():
            continue
        ds.append(f(y[idx], pa[idx]) - f(y[idx], pb[idx]))
    lo, hi = np.quantile(ds, [0.05, 0.95])
    return {"diff": float(point), "ci90": [float(lo), float(hi)], "n": int(len(y))}


def calibration(y, p, bins: int = 10) -> list[dict]:
    y, p = _clean(y, p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0, 1
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"p_mean": float(p[m].mean()), "y_rate": float(y[m].mean()), "n": int(m.sum())})
    return out
