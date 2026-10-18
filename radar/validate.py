"""Validation gates V0 / V1 / V2. Writes reports/validation.json and reports/validation.md.

V0  Audit of the original project design: does the ~98% classification accuracy
    measure prediction, or just reproduce K-Means labels from the same features?
V1  Price-only stress model vs GARCH(1,1), HAR-RV and India VIX, walk-forward.
V2  Do news features add out-of-sample skill over the price-only model?
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CFG, REPORT_DIR
from .data import provenance
from .metrics import calibration, paired_diff_ci, score
from .pipeline import VERSION

log = logging.getLogger("radar.validate")

MODEL_NAMES = {
    "p_model": "Regime Radar (HMM + logistic)",
    "p_gbm": "Gradient boosting (same features)",
    "p_vix": "India VIX only",
    "p_har": "HAR-RV",
    "p_garch": "GARCH(1,1)",
    "p_clim": "Base rate (climatology)",
}


def data_fingerprint(df: pd.DataFrame) -> str:
    import hashlib
    cols = ["close", "vix", "spx_ret_lag", "news_stress"]
    h = hashlib.sha1(pd.util.hash_pandas_object(df[cols].fillna(-9e9), index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def v0_audit(df: pd.DataFrame, seed: int = 0) -> dict:
    feats = ["ret", "rv22", "mom20", "ma50_dist", "dd252"]
    X = df[feats].dropna()
    Z = StandardScaler().fit_transform(X)                     # full-sample scaling, as in a typical WEKA flow
    km = KMeans(3, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_
    sizes = np.bincount(lab).tolist()

    cv = KFold(10, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
    rf_acc = cross_val_score(rf, Z, lab, cv=cv, scoring="accuracy").mean()
    sub = np.random.default_rng(seed).choice(len(Z), min(2500, len(Z)), replace=False)
    svm_acc = cross_val_score(SVC(), Z[sub], lab[sub], cv=KFold(5, shuffle=True, random_state=seed)).mean()

    # The honest version: predict the cluster label h days AHEAD, train on the past only
    h = CFG.horizon
    lab_s = pd.Series(lab, index=X.index)
    fut = lab_s.shift(-h).dropna().astype(int)
    Xf = pd.DataFrame(Z, index=X.index).loc[fut.index]
    split = int(len(Xf) * 0.7)
    rf2 = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(Xf.iloc[:split], fut.iloc[:split])
    pred = rf2.predict(Xf.iloc[split:])
    yt = fut.iloc[split:]
    persist = lab_s.loc[yt.index].to_numpy()                   # "the regime in h days = today's regime"
    return {
        "features": feats,
        "cluster_sizes": sizes,
        "shuffled_cv_accuracy": {"random_forest": float(rf_acc), "svm": float(svm_acc)},
        "future_label_chronological": {
            "horizon_days": h,
            "random_forest_accuracy": float(accuracy_score(yt, pred)),
            "random_forest_balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
            "persistence_accuracy": float(accuracy_score(yt, persist)),
            "persistence_balanced_accuracy": float(balanced_accuracy_score(yt, persist)),
        },
        "verdict": (
            "Shuffled cross-validation on K-Means labels mainly measures how well a classifier can redraw "
            "the K-Means boundaries from the same features, so very high accuracy is expected and is not "
            "evidence of forecasting. Predicting the label in the future with a chronological split, and "
            "comparing with the naive 'regime stays the same' rule, is the meaningful test."
        ),
    }


def _table(tl: pd.DataFrame, cols: list[str]) -> dict:
    return {c: score(tl["y"], tl[c]) for c in cols if c in tl}


def v1(tl: pd.DataFrame) -> dict:
    cols = ["p_model", "p_gbm", "p_vix", "p_har", "p_garch", "p_clim"]
    tab = _table(tl, cols)
    best_base = min(["p_har", "p_garch", "p_vix"], key=lambda c: tab[c].get("brier", 9))
    ci_har = paired_diff_ci(tl["y"], tl["p_model"], tl["p_har"], "brier")
    ci_best = paired_diff_ci(tl["y"], tl["p_model"], tl[best_base], "brier")
    beats_har = ci_har["ci90"][1] < 0
    beats_best = ci_best["ci90"][1] < 0
    ties_best = ci_best["ci90"][0] <= 0 <= ci_best["ci90"][1]
    if beats_best:
        verdict, status = "PASS — beats every baseline including the best one", "pass"
    elif beats_har and ties_best:
        verdict, status = f"PASS (qualified) — beats HAR-RV; statistically tied with {MODEL_NAMES[best_base]}", "pass"
    elif beats_har:
        verdict, status = f"PARTIAL — beats HAR-RV but loses to {MODEL_NAMES[best_base]}", "partial"
    else:
        verdict, status = "FAIL — does not beat HAR-RV; ship as a descriptive explainer only", "fail"
    by_year = []
    for yr, g in tl.groupby("fold"):
        s = {c: score(g["y"], g[c]).get("brier") for c in ["p_model", "p_vix", "p_har", "p_clim"]}
        s["year"] = int(yr)
        s["stress_rate"] = float(np.nanmean(g["y"]))
        by_year.append(s)
    return {"table": tab, "names": MODEL_NAMES, "best_baseline": best_base,
            "model_minus_har_brier": ci_har, "model_minus_best_brier": ci_best,
            "status": status, "verdict": verdict, "by_year": by_year,
            "calibration": calibration(tl["y"], tl["p_model"]),
            "period": [str(tl.index[0].date()), str(tl.index[-1].date())]}


def v2(tl_price: pd.DataFrame, tl_news: pd.DataFrame | None, news_days: int) -> dict:
    if tl_news is None or news_days < 500:
        return {"status": "insufficient", "news_days": news_days,
                "verdict": "NOT RUN — not enough news history. Run `python -m radar news-backfill` "
                           "(GDELT timelines go back to 2017) and re-run validation."}
    idx = tl_news.index.intersection(tl_price.index)
    a, b = tl_news.loc[idx], tl_price.loc[idx]
    y = a["y"]
    d_brier = paired_diff_ci(y, a["p_model"], b["p_model"], "brier")
    d_pr = paired_diff_ci(y, a["p_model"], b["p_model"], "pr_auc")
    if d_brier["ci90"][1] < 0:
        status, verdict = "pass", "PASS — news features improve out-of-sample stress forecasts (the 90% range of the Brier difference excludes zero)"
    elif d_brier["diff"] < 0:
        status, verdict = "weak", "INCONCLUSIVE — a small improvement, but the 90% range includes zero. News stays as context, not as a predictor."
    else:
        status, verdict = "fail", "FAIL — news does not improve forecasts. News stays in the UI as context, not as a predictor."
    return {"status": status, "verdict": verdict, "news_days": news_days,
            "period": [str(idx[0].date()), str(idx[-1].date())],
            "with_news": score(y, a["p_model"]), "price_only": score(y, b["p_model"]),
            "news_minus_price_brier": d_brier, "news_minus_price_pr_auc": d_pr}


def run_validation(mode: str, df: pd.DataFrame, tl_price: pd.DataFrame, tl_news: pd.DataFrame | None) -> dict:
    news_days = int(df["news_stress"].notna().sum()) if "news_stress" in df else 0
    rep = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "version": VERSION,
        "data_fingerprint": data_fingerprint(df),
        "provenance": provenance(mode),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.__dict__.items()},
        "target": f"1 if annualised realised vol over the next {CFG.horizon} sessions exceeds the "
                  f"{int(CFG.stress_quantile*100)}th percentile of the training window",
        "v0": v0_audit(df),
        "v1": v1(tl_price),
        "v2": v2(tl_price, tl_news, news_days),
    }
    rep["use_news_in_production"] = rep["v2"]["status"] == "pass"
    (REPORT_DIR / "validation.json").write_text(json.dumps(rep, indent=2, default=float))
    (REPORT_DIR / "validation.md").write_text(to_markdown(rep))
    return rep


def to_markdown(r: dict) -> str:
    v0, v1r, v2r = r["v0"], r["v1"], r["v2"]
    sim = r["mode"] == "demo"
    L = ["# Regime Radar — validation report", ""]
    if sim:
        L += ["> **SIMULATED DATA.** These numbers come from the built-in market simulator and only prove the "
              "pipeline works. Run `python -m radar validate --live` on your machine for real results.", ""]
    L += [f"Generated {r['generated_utc']} · source: {r['provenance']['source']}", "",
          f"**Target:** {r['target']}.", "",
          "## V0 — Audit of the original design", "",
          f"K-Means (k=3) cluster sizes: {v0['cluster_sizes']}", "",
          "| Test | Accuracy |", "|---|---:|",
          f"| Random Forest, shuffled 10-fold CV on same-day K-Means labels | {v0['shuffled_cv_accuracy']['random_forest']:.1%} |",
          f"| SVM, shuffled 5-fold CV on same-day K-Means labels | {v0['shuffled_cv_accuracy']['svm']:.1%} |",
          f"| Random Forest, label {v0['future_label_chronological']['horizon_days']} days ahead, chronological split | {v0['future_label_chronological']['random_forest_accuracy']:.1%} (balanced {v0['future_label_chronological']['random_forest_balanced_accuracy']:.1%}) |",
          f"| Naive persistence (label stays the same) | {v0['future_label_chronological']['persistence_accuracy']:.1%} (balanced {v0['future_label_chronological']['persistence_balanced_accuracy']:.1%}) |",
          "", v0["verdict"], "",
          "## V1 — Price-only model vs baselines (walk-forward, out-of-sample)", "",
          f"Period {v1r['period'][0]} → {v1r['period'][1]}", "",
          "| Model | PR-AUC | ROC-AUC | Brier | Brier skill |", "|---|---:|---:|---:|---:|"]
    for k, s in v1r["table"].items():
        if "brier" in s:
            L.append(f"| {v1r['names'][k]} | {s['pr_auc']:.3f} | {s['roc_auc']:.3f} | {s['brier']:.4f} | {s['brier_skill']:.3f} |")
    c = v1r["model_minus_best_brier"]
    L += ["", f"Model − best baseline ({v1r['names'][v1r['best_baseline']]}) Brier: {c['diff']:+.4f} "
              f"(90% block-bootstrap CI {c['ci90'][0]:+.4f} to {c['ci90'][1]:+.4f}; negative = model better)", "",
          f"**Verdict: {v1r['verdict']}**", "", "## V2 — Does news add skill?", "", f"**Verdict: {v2r['verdict']}**", ""]
    if v2r.get("status") not in (None, "insufficient"):
        d = v2r["news_minus_price_brier"]
        L += [f"Period {v2r['period'][0]} → {v2r['period'][1]}; Brier with news − without: {d['diff']:+.4f} "
              f"(90% CI {d['ci90'][0]:+.4f} to {d['ci90'][1]:+.4f})", ""]
    L += ["## Production decision", "",
          f"News features in the production model: **{'ON' if r['use_news_in_production'] else 'OFF'}** "
          "(switched on only if V2 passes).", ""]
    return "\n".join(L)
