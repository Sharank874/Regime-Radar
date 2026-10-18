"""Engine: builds features, runs the walk-forward models and serves views to the API."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import ledger
from .config import CFG, FEATURE_LABELS, REPORT_DIR, STATE_NAMES
from .data import load_market, provenance
from .features import add_target, build_features
from .news import category_snapshot, collection_status, demo_headlines, event_impact, recent_headlines
from .overnight import overnight
from .pipeline import VERSION, score_rows, walk_forward
from .validate import data_fingerprint
from .taxonomy import CATEGORIES

log = logging.getLogger("radar.engine")


def _f(x):
    """JSON-safe float."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) or np.isinf(x) else round(x, 6)


def level_for(p: float, base: float) -> str:
    r = p / max(base, 1e-6)
    return "Low" if r < 0.75 else "Moderate" if r < 1.25 else "Elevated" if r < 2.0 else "High"


class Engine:
    def __init__(self, mode: str = "demo"):
        self.mode = mode
        self.built_utc = None
        self.provisional: dict | None = None

    # ------------------------------------------------------------------ build
    def build(self, refresh: bool = False, validate: bool = False):
        self.data = load_market(self.mode, refresh=refresh)
        self.df = add_target(build_features(self.data), CFG.horizon)
        self.news_days = int(self.df["news_stress"].notna().sum())
        log.info("features: %d rows, news days: %d", len(self.df), self.news_days)
        self.res_price = walk_forward(self.df, CFG, use_news=False)
        self.res_news = walk_forward(self.df, CFG, use_news=True) if self.news_days >= 500 else None

        rep_path = REPORT_DIR / "validation.json"
        self.fingerprint = data_fingerprint(self.df)
        stale = (not rep_path.exists()) or any(json.loads(rep_path.read_text()).get(k) != v for k, v in
                                               (("mode", self.mode), ("version", VERSION),
                                                ("data_fingerprint", self.fingerprint)))
        if validate or stale:
            from .validate import run_validation
            run_validation(self.mode, self.df, self.res_price["timeline"],
                           self.res_news["timeline"] if self.res_news else None)
        self.report = json.loads(rep_path.read_text())
        self.use_news = bool(self.report.get("use_news_in_production")) and self.res_news is not None
        self._assemble()
        self.built_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def _assemble(self):
        """Production timeline: news model where it exists (if V2 passed), else price-only."""
        tp = self.res_price["timeline"]
        self.attr = {}
        self._fold_for = {}
        if self.use_news:
            tn = self.res_news["timeline"]
            tl = pd.concat([tp[tp.index < tn.index[0]], tn])
            for i, d in enumerate(tp.index):
                if d < tn.index[0]:
                    self.attr[d] = ("price", i)
            for i, d in enumerate(tn.index):
                self.attr[d] = ("news", i)
        else:
            tl = tp
            self.attr = {d: ("price", i) for i, d in enumerate(tp.index)}
        tl = tl.copy()
        tl["regime"] = tl[[f"post_{s}" for s in range(CFG.n_states)]].to_numpy().argmax(1)
        self.tl = tl
        self.regime_names = pd.Series([STATE_NAMES[i] for i in tl["regime"]], index=tl.index)
        ledger.sync_backtest(self.mode, tl, VERSION, self.regime_names)
        ledger.score_live(self.mode, self.df)
        if self.mode == "live":
            last = tl.index[-1]
            ledger.record_live(self.mode, last, tl.loc[last], self.regime_names[last], VERSION)
        self.impact = event_impact(self.df)

    def _res(self, kind: str):
        return self.res_news if kind == "news" else self.res_price

    def latest_fold(self):
        res = self.res_news if self.use_news else self.res_price
        return res["folds"][max(res["folds"])]

    # ------------------------------------------------------------------ nowcast (live, intraday)
    def nowcast(self, intraday: dict) -> dict | None:
        """Provisional update from a delayed intraday print (live mode, market hours)."""
        if "nifty" not in intraday:
            return None
        ts = pd.Timestamp(intraday["nifty"]["time"])
        day = ts.tz_convert("Asia/Kolkata").normalize().tz_localize(None) if ts.tzinfo else ts.normalize()
        if day <= self.df.index[-1]:
            return None
        data = {k: v.copy() for k, v in self.data.items()}
        px = intraday["nifty"]["price"]
        data["nifty"] = pd.concat([data["nifty"], pd.DataFrame(
            {"open": [px], "high": [px], "low": [px], "close": [px], "volume": [0]}, index=[day])])
        if "india_vix" in intraday:
            data["india_vix"] = pd.concat([data["india_vix"], pd.DataFrame(
                {"close": [intraday["india_vix"]["price"]]}, index=[day])])
        df = add_target(build_features(data), CFG.horizon)
        fold = self.latest_fold()
        sc, attr = score_rows(fold, df, pd.Index([day]))
        self.provisional = {"date": day, "row": sc.iloc[0], "attr": {k: v[0] for k, v in attr.items()},
                            "feat": df.loc[day], "as_of_print": intraday["nifty"]["time"], "fold": fold}
        return self.snapshot(None, provisional=True)

    # ------------------------------------------------------------------ views
    def dates(self) -> list[str]:
        return [str(d.date()) for d in self.tl.index]

    def _resolve(self, date) -> pd.Timestamp:
        if date is None:
            return self.tl.index[-1]
        d = pd.Timestamp(date)
        pos = self.tl.index.searchsorted(d, side="right") - 1
        return self.tl.index[max(pos, 0)]

    def _row(self, d):
        return self.tl.loc[d], self.df.loc[d]

    def snapshot(self, date=None, provisional: bool = False) -> dict:
        if provisional and self.provisional:
            d, r, f = self.provisional["date"], self.provisional["row"], self.provisional["feat"]
        else:
            d = self._resolve(date)
            r, f = self._row(d)
        post = [_f(r[f"post_{s}"]) for s in range(CFG.n_states)]
        prior = [_f(r[f"prior_{s}"]) for s in range(CFG.n_states)]
        base = float(r["p_clim"])
        p = float(r["p_model"]) if not np.isnan(r["p_model"]) else np.nan
        k = int(np.nanargmax(post)) if any(v is not None for v in post) else 0
        i = self.tl.index.get_indexer([d])[0] if d in self.tl.index else len(self.tl)
        prev_p = float(self.tl["p_model"].iloc[i - 1]) if i > 0 else np.nan
        return {
            "date": str(d.date()), "provisional": provisional, "mode": self.mode,
            "as_of_print": self.provisional["as_of_print"] if provisional else None,
            "market": {"close": _f(f["close"]), "ret": _f(f["ret"]), "rv5": _f(f["rv5"]), "rv22": _f(f["rv22"]),
                       "vix": _f(f["vix"]), "dd252": _f(f["dd252"]), "spx_ret_lag": _f(f["spx_ret_lag"])},
            "regime": {"names": STATE_NAMES, "post": post, "prior": prior, "label": STATE_NAMES[k],
                       "confidence": post[k]},
            "stress": {"p": _f(p), "lo": _f(r["p_lo"]), "hi": _f(r["p_hi"]), "base_rate": _f(base),
                       "level": level_for(p, base) if not np.isnan(p) else None,
                       "ratio": _f(p / base) if not np.isnan(p) else None, "prev": _f(prev_p),
                       "threshold_vol": _f(r["threshold"]), "horizon": CFG.horizon},
            "baselines": {"vix": _f(r["p_vix"]), "har": _f(r["p_har"]), "garch": _f(r["p_garch"]), "gbm": _f(r["p_gbm"])},
            "outcome": {"y": _f(r["y"]), "fwd_rv": _f(f.get("fwd_rv"))},
            "fold": int(r["fold"]),
        }

    def explain(self, date=None, provisional: bool = False) -> dict:
        if provisional and self.provisional:
            d, r, f, a = (self.provisional["date"], self.provisional["row"], self.provisional["feat"],
                          self.provisional["attr"])
            fold = self.provisional["fold"]
            phi, L, prev = a["phi"], a["L"], a["prev_post"]
        else:
            d = self._resolve(date)
            r, f = self._row(d)
            kind, i = self.attr[d]
            res = self._res(kind)
            phi, L, prev = res["phi"][i], res["L"][i], res["prev_post"][i]
            fold = res["folds"][int(r["fold"])]
        S = CFG.n_states
        prior = np.array([r[f"prior_{s}"] for s in range(S)])
        post = np.array([r[f"post_{s}"] for s in range(S)])
        feats = []
        for j, col in enumerate(fold.hmm_cols):
            raw = f.get(col)
            z = (raw - fold.hmm_mu[col]) / fold.hmm_sd[col] if raw is not None and not pd.isna(raw) else None
            feats.append({"key": col, "label": FEATURE_LABELS.get(col, col), "value": _f(raw), "z": _f(z),
                          "phi": [_f(x) for x in phi[:, j]], "loglik": [_f(x) for x in L[:, j]],
                          "used": z is not None})
        contribs = []
        for c in fold.clf_cols:
            val = r[f"post_{S-1}"] if c == "p_stress" else r["post_0"] if c == "p_calm" else f.get(c)
            contribs.append({"key": c, "label": FEATURE_LABELS.get(c, c), "value": _f(val),
                             "contrib": _f(r.get(f"c_{c}"))})
        contribs.sort(key=lambda x: -abs(x["contrib"] or 0))
        logit = float(r["c_intercept"]) + float(np.nansum([c["contrib"] or 0 for c in contribs]))
        news_ctx = category_snapshot(f)
        # day-over-day change of the stress probability, attributed to features
        if provisional and self.provisional:
            prev_row = self.tl.iloc[-1]
        else:
            i = self.tl.index.get_loc(d)
            prev_row = self.tl.iloc[i - 1] if i > 0 else None
        change = self._change(prev_row, r, fold) if prev_row is not None else None
        return {
            "date": str(d.date()), "state_names": STATE_NAMES,
            "hmm": {"prev_post": [_f(x) for x in prev], "prior": [_f(x) for x in prior], "post": [_f(x) for x in post],
                    "drift": [_f(prior[s] - prev[s]) for s in range(S)] if not np.isnan(prev).any() else None,
                    "features": feats, "news_in_model": "news_stress" in fold.hmm_cols},
            "clf": {"intercept": _f(r["c_intercept"]), "logit": _f(logit), "p": _f(r["p_model"]),
                    "contributions": contribs, "news_in_model": fold.news_in_clf},
            "news_context": news_ctx,
            "change": change,
        }

    @staticmethod
    def _change(prev, cur, fold) -> dict | None:
        """Exact split of Δlogit into feature contributions; Δp allocated proportionally."""
        if pd.isna(prev["p_model"]) or pd.isna(cur["p_model"]):
            return None
        lg = lambda p: np.log(p / (1 - p))
        p0, p1 = float(prev["p_model"]), float(cur["p_model"])
        dl_total = lg(p1) - lg(p0)
        items = []
        refit = int(prev["fold"]) != int(cur["fold"])
        acc = 0.0
        for c in fold.clf_cols:
            a, b = prev.get(f"c_{c}", np.nan), cur.get(f"c_{c}", np.nan)
            if pd.isna(a) or pd.isna(b):
                continue
            dl = float(b - a)
            acc += dl
            items.append({"key": c, "label": FEATURE_LABELS.get(c, c), "dlogit": dl})
        resid = dl_total - acc                       # intercept change at a yearly refit
        if abs(resid) > 1e-9:
            items.append({"key": "refit", "label": "Yearly model refit", "dlogit": resid})
        dp = p1 - p0
        for it in items:
            it["dp"] = dp * it["dlogit"] / dl_total if abs(dl_total) > 1e-12 else 0.0
        items.sort(key=lambda x: -abs(x["dp"]))
        return {"p_prev": p0, "p_now": p1, "dp": dp, "dlogit": dl_total, "refit": refit, "items": items}

    def moves(self, n: int = 40, end=None) -> list[dict]:
        d = self._resolve(end)
        i = self.tl.index.get_loc(d)
        out = []
        for j in range(max(1, i - n + 1), i + 1):
            prev, cur = self.tl.iloc[j - 1], self.tl.iloc[j]
            fold = self._res(self.attr[self.tl.index[j]][0])["folds"][int(cur["fold"])]
            ch = self._change(prev, cur, fold)
            if ch is None:
                continue
            top = ch["items"][0] if ch["items"] else None
            out.append({"date": str(self.tl.index[j].date()), "p": _f(ch["p_now"]), "dp": _f(ch["dp"]),
                        "top": top["label"] if top else None, "top_dp": _f(top["dp"]) if top else None,
                        "regime": STATE_NAMES[int(cur["regime"])]})
        return out[::-1]

    def timeline(self, start=None, end=None) -> dict:
        tl = self.tl
        if start:
            tl = tl[tl.index >= pd.Timestamp(start)]
        if end:
            tl = tl[tl.index <= pd.Timestamp(end)]
        f = self.df.loc[tl.index]
        S = CFG.n_states
        return {
            "dates": [str(d.date()) for d in tl.index],
            "close": [_f(x) for x in f["close"]],
            "vix": [_f(x) for x in f["vix"]],
            "post": [[_f(x) for x in tl[f"post_{s}"]] for s in range(S)],
            "p": [_f(x) for x in tl["p_model"]],
            "lo": [_f(x) for x in tl["p_lo"]],
            "hi": [_f(x) for x in tl["p_hi"]],
            "p_vix": [_f(x) for x in tl["p_vix"]],
            "y": [_f(x) for x in tl["y"]],
            "fwd_rv": [_f(x) for x in f["fwd_rv"]],
            "rv5": [_f(x) for x in f["rv5"]],
            "regime": [int(x) for x in tl["regime"]],
            "base_rate": [_f(x) for x in tl["p_clim"]],
        }

    def model_info(self) -> dict:
        fold = self.latest_fold()
        p = fold.hmm.params
        S = CFG.n_states
        cols = fold.hmm_cols
        emis = []
        for j, c in enumerate(cols):
            m, s = float(fold.hmm_mu[c]), float(fold.hmm_sd[c])
            emis.append({"key": c, "label": FEATURE_LABELS.get(c, c),
                         "mean": [p.mu[k, j] * s + m for k in range(S)],
                         "std": [np.sqrt(p.var[k, j]) * s for k in range(S)],
                         "mean_z": [float(p.mu[k, j]) for k in range(S)], "std_z": [float(np.sqrt(p.var[k, j])) for k in range(S)],
                         "center": m, "scale": s})
        tl, f = self.tl, self.df.loc[self.tl.index]
        stats = []
        for k in range(S):
            m = tl["regime"] == k
            runs = (m != m.shift()).cumsum()[m]
            dur = runs.value_counts().mean() if m.any() else np.nan
            stats.append({"name": STATE_NAMES[k], "occupancy": _f(m.mean()), "avg_run_days": _f(dur),
                          "expected_duration": _f(1 / (1 - p.A[k, k])),
                          "mean_daily_ret": _f(f.loc[m, "ret"].mean()), "ann_vol": _f(f.loc[m, "ret"].std() * np.sqrt(252)),
                          "stress_week_rate": _f(tl.loc[m, "y"].mean())})
        coef = fold.clf[1].coef_[0]
        return {
            "version": VERSION, "fold_year": fold.year, "train_end": str(fold.train_end.date()),
            "state_names": STATE_NAMES, "transition": [[_f(x) for x in row] for row in p.A],
            "emissions": json.loads(json.dumps(emis, default=float)), "state_stats": stats,
            "classifier": {"features": [{"key": c, "label": FEATURE_LABELS.get(c, c), "coef": _f(w)}
                                        for c, w in zip(fold.clf_cols, coef)],
                           "intercept": _f(fold.clf[1].intercept_[0])},
            "folds": [res.meta | {"year": y} for y, res in (self.res_news if self.use_news else self.res_price)["folds"].items()],
            "config": {"horizon": CFG.horizon, "stress_quantile": CFG.stress_quantile, "n_states": S,
                       "hmm_features": cols, "sticky": CFG.hmm_sticky},
            "use_news": self.use_news,
        }

    def news(self, date=None) -> dict:
        d = self._resolve(date)
        f = self.df.loc[d]
        if self.mode == "demo":
            heads = []
            i = self.df.index.get_loc(d)
            for back in range(0, 4):
                dd = self.df.index[i - back]
                heads += demo_headlines(dd, self.df.loc[dd])
            seen, uniq = set(), []
            for h in heads:
                if h["title"] not in seen:
                    seen.add(h["title"])
                    uniq.append(h)
            heads = uniq
        else:
            # NSE closes 15:30 IST = 10:00 UTC. For the latest session show everything since 3 days
            # before its close up to now (news after the close matters for the next session);
            # for a past session show the 3 days up to its close.
            close_utc = (d.tz_localize("UTC") + pd.Timedelta(hours=10)).to_pydatetime()
            latest = d == self.tl.index[-1]
            heads = recent_headlines(since=close_utc - pd.Timedelta(hours=72).to_pytimedelta(),
                                     until=None if latest else close_utc)
        win = self.df.loc[:d].tail(60)
        heat = {"dates": [str(x.date()) for x in win.index],
                "rows": [{"key": c.key, "label": c.label, "color": c.color,
                          "z": [_f(v) for v in win.get(f"surprise_{c.key}", pd.Series(np.nan, index=win.index))]}
                         for c in CATEGORIES]}
        return {"date": str(d.date()), "headlines": heads[:25], "categories": category_snapshot(f),
                "news_stress": _f(f.get("news_stress")), "impact": self.impact, "heatmap": heat,
                "use_news_in_model": self.use_news, "news_days": self.news_days,
                "collection": collection_status() if self.mode == "live" else None}

    def overnight(self, date=None) -> dict:
        d = self._resolve(date)
        return overnight(self.df, self.data["spx"], self.data["us_vix"], d)

    def track_record(self) -> dict:
        from .metrics import calibration, score
        bt = ledger.read(self.mode, "backtest")
        live = ledger.read(self.mode, "live")
        out = {"backtest": {"n": len(bt)}, "live": {"n": len(live)}}
        if len(bt):
            sc = bt.dropna(subset=["outcome"])
            out["backtest"].update(score(sc["outcome"], sc["p"]))
            out["backtest"]["calibration"] = calibration(sc["outcome"], sc["p"])
            out["backtest"]["recent"] = bt.tail(30).replace({np.nan: None}).to_dict("records")
        if len(live):
            sc = live.dropna(subset=["outcome"])
            out["live"]["scored"] = int(len(sc))
            if len(sc) >= 30:
                out["live"].update(score(sc["outcome"], sc["p"]))
            out["live"]["rows"] = live.tail(60).replace({np.nan: None}).to_dict("records")
        out["validation"] = self.report
        return out

    def meta(self) -> dict:
        return {"mode": self.mode, "version": VERSION, "built_utc": self.built_utc,
                "first_date": self.dates()[0], "last_date": self.dates()[-1],
                "provenance": provenance(self.mode), "use_news": self.use_news, "news_days": self.news_days,
                "state_names": STATE_NAMES, "horizon": CFG.horizon,
                "v1": self.report["v1"]["status"], "v2": self.report["v2"]["status"],
                "episodes": self.episodes()}

    def episodes(self, k: int = 6) -> list[dict]:
        """Top stress episodes (for the replay time machine): peaks of 20-day realised vol."""
        f = self.df.loc[self.tl.index, "rv22"].dropna()
        out, used = [], []
        for d, v in f.sort_values(ascending=False).items():
            if any(abs((d - u).days) < 180 for u in used):
                continue
            used.append(d)
            pos = self.tl.index.get_loc(d)
            start = self.tl.index[max(pos - 45, 0)]
            end = self.tl.index[min(pos + 25, len(self.tl) - 1)]
            out.append({"peak": str(d.date()), "start": str(start.date()), "end": str(end.date()), "rv22": _f(v)})
            if len(out) >= k:
                break
        return sorted(out, key=lambda x: x["peak"])
