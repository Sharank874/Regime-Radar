"""Walk-forward engine: fits every model on data strictly before each test year,
then scores that year. Produces one out-of-sample row per trading day.

Fold for test year Y
  train rows : first usable date .. last date before Y-01-01, minus `horizon` days (purge,
               so training targets never overlap the test year)
  test rows  : every trading day in year Y
  threshold  : 80th percentile of forward 5-day realised vol on the TRAIN rows only
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CACHE_DIR, CFG, ModelConfig
from .hmm import GaussianHMM, effective_temper, shapley_update

log = logging.getLogger("radar.pipeline")
VERSION = "rr-1.1"


# ----------------------------------------------------------------------------- helpers
def block_bootstrap_idx(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


class Logit1D:
    """Logistic map from a single score to a probability (used by baselines)."""

    def __init__(self, cols):
        self.cols = cols
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))

    def fit(self, X: pd.DataFrame, y):
        self.m.fit(X[self.cols].to_numpy(), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[self.cols].to_numpy()
        out = np.full(len(X), np.nan)
        ok = ~np.isnan(Z).any(1)
        if ok.any():
            out[ok] = self.m.predict_proba(Z[ok])[:, 1]
        return out


class Garch11:
    """GARCH(1,1) fitted with `arch` on training returns; variance filtered forward
    with fixed parameters; mean 5-day variance forecast as the risk score."""

    def __init__(self, horizon: int):
        self.h = horizon

    def fit(self, r_train: pd.Series):
        from arch import arch_model
        res = arch_model(r_train.dropna() * 100, vol="GARCH", p=1, q=1, mean="Constant", dist="normal").fit(disp="off")
        p = res.params
        self.mu, self.w, self.a, self.b = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
        self.s2_0 = float(np.var(r_train.dropna() * 100))
        return self

    def forecast(self, r_all: pd.Series) -> pd.Series:
        e = (r_all.fillna(0).to_numpy() * 100) - self.mu
        s2 = np.empty(len(e))
        s2[0] = self.s2_0
        for t in range(1, len(e)):
            s2[t] = self.w + self.a * e[t - 1] ** 2 + self.b * s2[t - 1]
        # one-step-ahead variance for t+1 given info at t
        nxt = self.w + self.a * e ** 2 + self.b * s2
        persist = self.a + self.b
        lr = self.w / max(1 - persist, 1e-6)
        f = np.mean([lr + persist ** (k) * (nxt - lr) for k in range(self.h)], axis=0)
        return pd.Series(np.log(np.maximum(f, 1e-8)), index=r_all.index)


@dataclass
class Fold:
    year: int
    train_end: pd.Timestamp
    threshold: float
    hmm: GaussianHMM
    hmm_cols: list
    hmm_mu: pd.Series
    hmm_sd: pd.Series
    clf: object
    clf_cols: list
    boot: list
    gbm: object
    baselines: dict
    base_rate: float
    news_in_clf: bool
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- core
def _hmm_inputs(df: pd.DataFrame, cols: list, mu=None, sd=None):
    X = df[cols]
    if mu is None:
        mu, sd = X.mean(), X.std().replace(0, 1)
    return ((X - mu) / sd).to_numpy(), mu, sd


def _hmm_frame(fold: Fold, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Causal filter over `df` (from the start) with this fold's parameters."""
    Z, _, _ = _hmm_inputs(df, fold.hmm_cols, fold.hmm_mu, fold.hmm_sd)
    prior, post, L = fold.hmm.filter(Z)
    phi = shapley_update(prior, L)
    out = pd.DataFrame(index=df.index)
    S = prior.shape[1]
    for s in range(S):
        out[f"prior_{s}"] = prior[:, s]
        out[f"post_{s}"] = post[:, s]
    out["p_stress"] = post[:, S - 1]
    out["p_calm"] = post[:, 0]
    return out, {"prior": prior, "post": post, "L": L, "phi": phi}


def _core_ok(df: pd.DataFrame, cfg: ModelConfig) -> pd.Series:
    return df[list(cfg.hmm_features)].notna().all(axis=1)


def _fit_fold(df: pd.DataFrame, year: int, cfg: ModelConfig, use_news: bool) -> Fold:
    start_test = pd.Timestamp(f"{year}-01-01")
    pre = df.index[df.index < start_test]
    train_end = pre[-(cfg.horizon + 1)]             # purge: last train target ends before the test year
    ok = _core_ok(df, cfg)
    tr = df.loc[(df.index <= train_end) & ok]

    hmm_cols = list(cfg.hmm_features) + ([cfg.hmm_news_feature] if use_news else [])
    Z, mu, sd = _hmm_inputs(tr, hmm_cols)
    hmm = GaussianHMM(cfg.n_states, cfg.hmm_iter, n_init=cfg.hmm_inits, sticky=cfg.hmm_sticky,
                      order_by=1, seed=cfg.seed, temper=effective_temper(Z)).fit(Z)

    fold = Fold(year, train_end, np.nan, hmm, hmm_cols, mu, sd, None, [], [], None, {}, np.nan, use_news)
    hm, _ = _hmm_frame(fold, df.loc[ok & (df.index <= train_end)])
    trX = tr.join(hm[["p_stress", "p_calm"]])
    trX = trX[trX["fwd_rv"].notna()]
    thr = float(trX["fwd_rv"].quantile(cfg.stress_quantile))
    y = (trX["fwd_rv"] > thr).astype(int).to_numpy()

    clf_cols = list(cfg.clf_features) + (list(cfg.clf_news_features) if use_news else [])
    mask = trX[clf_cols].notna().all(axis=1).to_numpy()
    Xc, yc = trX.loc[mask, clf_cols].to_numpy(), y[mask]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg.clf_C, max_iter=2000)).fit(Xc, yc)

    rng = np.random.default_rng(cfg.seed + year)
    boot = []
    for _ in range(cfg.n_bootstrap):
        idx = block_bootstrap_idx(len(Xc), cfg.block_len, rng)
        if yc[idx].min() == yc[idx].max():
            continue
        boot.append(make_pipeline(StandardScaler(), LogisticRegression(C=cfg.clf_C, max_iter=2000)).fit(Xc[idx], yc[idx]))

    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                         l2_regularization=1.0, random_state=cfg.seed).fit(Xc, yc)

    trB = trX.copy()
    trB["log_rv1"] = np.log(trB["ret"].abs() + 1e-4)
    baselines = {
        "har": Logit1D(["log_rv1", "log_rv5", "log_rv22"]),
        "vix": Logit1D(["log_vix"]),
    }
    for b in baselines.values():
        mb = trB[b.cols].notna().all(axis=1).to_numpy()
        b.fit(trB.loc[mb], y[mb])
    g = Garch11(cfg.horizon).fit(df.loc[df.index <= train_end, "ret"])
    gf = g.forecast(df["ret"]).rename("garch_f")
    trG = trB.join(gf)
    lg = Logit1D(["garch_f"])
    mg = trG["garch_f"].notna().to_numpy()
    lg.fit(trG.loc[mg], y[mg])
    baselines["garch"] = (g, lg)

    fold.threshold, fold.clf, fold.clf_cols, fold.boot, fold.gbm = thr, clf, clf_cols, boot, gbm
    fold.baselines, fold.base_rate = baselines, float(y.mean())
    fold.meta = {"n_train": int(len(yc)), "train_start": str(trX.index[0].date()),
                 "train_end": str(train_end.date()), "threshold_ann_vol": thr, "base_rate": float(y.mean())}
    return fold


def score_rows(fold: Fold, df: pd.DataFrame, rows: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Score `rows` with a fitted fold. `df` must contain history from the start
    (the HMM filter is run causally over it)."""
    ok = _core_ok(df, CFG)
    dfo = df.loc[ok]
    hm, raw = _hmm_frame(fold, dfo)
    X = df.loc[rows].join(hm, how="left")
    X["log_rv1"] = np.log(X["ret"].abs() + 1e-4)
    Xc = X[fold.clf_cols].to_numpy()
    good = ~np.isnan(Xc).any(1)
    p = np.full(len(X), np.nan)
    p10 = np.full(len(X), np.nan)
    p90 = np.full(len(X), np.nan)
    pg = np.full(len(X), np.nan)
    contrib = np.full((len(X), len(fold.clf_cols)), np.nan)
    if good.any():
        p[good] = fold.clf.predict_proba(Xc[good])[:, 1]
        pg[good] = fold.gbm.predict_proba(Xc[good])[:, 1]
        if fold.boot:
            B = np.stack([m.predict_proba(Xc[good])[:, 1] for m in fold.boot])
            p10[good], p90[good] = np.quantile(B, 0.1, 0), np.quantile(B, 0.9, 0)
        sc, lr = fold.clf[0], fold.clf[1]
        z = (Xc[good] - sc.mean_) / sc.scale_
        contrib[good] = z * lr.coef_[0]
    out = pd.DataFrame(index=rows)
    for s in range(fold.hmm.S):
        out[f"prior_{s}"] = X[f"prior_{s}"].to_numpy()
        out[f"post_{s}"] = X[f"post_{s}"].to_numpy()
    out["p_model"] = p
    out["p_lo"], out["p_hi"] = p10, p90
    out["p_gbm"] = pg
    out["p_har"] = fold.baselines["har"].predict(X)
    out["p_vix"] = fold.baselines["vix"].predict(X)
    g, lg = fold.baselines["garch"]
    X["garch_f"] = g.forecast(df["ret"]).reindex(X.index)
    out["p_garch"] = lg.predict(X)
    out["p_clim"] = fold.base_rate
    out["threshold"] = fold.threshold
    out["fold"] = fold.year
    out["y"] = np.where(df.loc[rows, "fwd_rv"].notna(), (df.loc[rows, "fwd_rv"] > fold.threshold).astype(float), np.nan)
    for j, c in enumerate(fold.clf_cols):
        out[f"c_{c}"] = contrib[:, j]
    out["c_intercept"] = float(fold.clf[1].intercept_[0])

    # per-row HMM attribution (aligned to rows)
    pos = dfo.index.get_indexer(rows)
    have = pos >= 0
    S, K = raw["L"].shape[1], raw["L"].shape[2]
    phi = np.full((len(rows), S, K), np.nan)
    L = np.full((len(rows), S, K), np.nan)
    phi[have] = raw["phi"][pos[have]]
    L[have] = raw["L"][pos[have]]
    prev_post = np.full((len(rows), S), np.nan)
    pp = np.vstack([raw["post"][:1] * np.nan, raw["post"][:-1]])
    prev_post[have] = pp[pos[have]]
    return out, {"phi": phi, "L": L, "prev_post": prev_post}


def _cache_key(df: pd.DataFrame, year: int, cfg: ModelConfig, use_news: bool) -> str:
    cut = df.index[df.index < pd.Timestamp(f"{year}-01-01")]
    sub = df.loc[cut].drop(columns=["fwd_rv", "fwd_absret1"], errors="ignore")
    h = hashlib.sha1(pd.util.hash_pandas_object(sub.fillna(-9e9), index=True).to_numpy().tobytes())
    h.update(repr((cfg, use_news, VERSION)).encode())
    return h.hexdigest()[:16]


def get_fold(df, year, cfg=CFG, use_news=False) -> Fold:
    key = _cache_key(df, year, cfg, use_news)
    path = CACHE_DIR / f"fold_{year}_{int(use_news)}_{key}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    fold = _fit_fold(df, year, cfg, use_news)
    with open(path, "wb") as fh:
        pickle.dump(fold, fh)
    return fold


def walk_forward(df: pd.DataFrame, cfg: ModelConfig = CFG, use_news: bool = False,
                 first_year: int | None = None) -> dict:
    """Returns {'timeline': DataFrame, 'attr': {...}, 'folds': {year: Fold}}"""
    first_year = first_year or (max(cfg.first_test_year, cfg.news_first_year + 2) if use_news else cfg.first_test_year)
    last_year = df.index[-1].year
    parts, phis, Ls, prevs, folds = [], [], [], [], {}
    for year in range(first_year, last_year + 1):
        rows = df.index[(df.index.year == year)]
        if not len(rows):
            continue
        fold = get_fold(df, year, cfg, use_news)
        folds[year] = fold
        sc, attr = score_rows(fold, df, rows)
        parts.append(sc)
        phis.append(attr["phi"])
        Ls.append(attr["L"])
        prevs.append(attr["prev_post"])
        log.info("fold %s: train->%s thr=%.3f n=%d", year, fold.train_end.date(), fold.threshold, fold.meta["n_train"])
    tl = pd.concat(parts)
    return {"timeline": tl, "phi": np.concatenate(phis), "L": np.concatenate(Ls),
            "prev_post": np.concatenate(prevs), "folds": folds}
