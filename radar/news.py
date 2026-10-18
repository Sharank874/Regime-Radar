"""News engine: GDELT DOC 2.0 (daily category timelines since 2017 + recent article lists)
and RSS feeds, tagged with the India event taxonomy and stored in SQLite.

GDELT notes
  * timeline modes (timelinevolraw / timelinetone) cover 2017-01-01 → now
  * article lists cover only the most recent ~3 months
  * be polite: one request every ~5 seconds
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_DIR
from .taxonomy import CATEGORIES, CAT_BY_KEY, headline_tone, tag_headline

log = logging.getLogger("radar.news")
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
DB = DATA_DIR / "news.sqlite"
NEWS_DAILY = RAW_DIR / "news_daily.csv"

RSS_FEEDS = {
    "RBI press releases": "https://www.rbi.org.in/pressreleases_rss.xml",
    "ET Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Livemint Markets": "https://www.livemint.com/rss/markets",
}


# ----------------------------------------------------------------------------- storage
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS headlines(
        id TEXT PRIMARY KEY, published_utc TEXT, title TEXT, url TEXT, source TEXT,
        categories TEXT, tone REAL, origin TEXT)""")
    return con


def store(items: list[dict]) -> int:
    con = _db()
    n = 0
    for it in items:
        hid = hashlib.sha1((it["title"].strip().lower()).encode()).hexdigest()[:16]
        cur = con.execute("INSERT OR IGNORE INTO headlines VALUES (?,?,?,?,?,?,?,?)",
                          (hid, it["published_utc"], it["title"], it.get("url", ""), it.get("source", ""),
                           ",".join(it["categories"]), it["tone"], it.get("origin", "")))
        n += cur.rowcount
    con.commit()
    con.close()
    return n


STATUS = DATA_DIR / "news_status.json"


def collect() -> dict:
    """Fetch RSS + GDELT article lists, store them, and record what happened (shown in the UI)."""
    import json
    status = {"last_run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": {}}
    items = []
    for name, fn in (("rss", rss_headlines), ("gdelt", lambda: gdelt_articles(24))):
        try:
            got = fn()
            items += got
            status["sources"][name] = {"fetched": len(got)}
        except Exception as e:
            status["sources"][name] = {"fetched": 0, "error": str(e)[:200]}
    status["new_stored"] = store(items)
    con = _db()
    status["total_stored"] = con.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    con.close()
    STATUS.write_text(json.dumps(status, indent=2))
    return status


def collection_status() -> dict | None:
    import json
    return json.loads(STATUS.read_text()) if STATUS.exists() else None


def recent_headlines(until: datetime | None = None, hours: int = 72, limit: int = 60,
                     since: datetime | None = None) -> list[dict]:
    con = _db()
    until = until or datetime.now(timezone.utc)
    since = since or (until - timedelta(hours=hours))
    rows = con.execute("SELECT published_utc,title,url,source,categories,tone,origin FROM headlines "
                       "WHERE published_utc>=? AND published_utc<=? ORDER BY published_utc DESC LIMIT ?",
                       (since.isoformat(), until.isoformat(), limit)).fetchall()
    con.close()
    return [{"published_utc": r[0], "title": r[1], "url": r[2], "source": r[3],
             "categories": r[4].split(","), "tone": r[5], "origin": r[6]} for r in rows]


def _item(title, url, source, published, origin) -> dict:
    return {"title": title, "url": url, "source": source, "published_utc": published,
            "categories": tag_headline(title), "tone": headline_tone(title), "origin": origin}


# ----------------------------------------------------------------------------- GDELT
def _get(params: dict, retries: int = 3) -> dict | None:
    import httpx
    for k in range(retries):
        try:
            r = httpx.get(GDELT, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (k + 1))
                continue
            r.raise_for_status()
            return r.json() if r.text.strip().startswith("{") else None
        except Exception as e:
            log.warning("GDELT request failed (%s): %s", params.get("mode"), e)
            time.sleep(5 * (k + 1))
    return None


def gdelt_articles(hours: int = 24) -> list[dict]:
    out = []
    for c in CATEGORIES:
        js = _get({"query": f"{c.gdelt_query} sourcelang:english", "mode": "artlist", "maxrecords": 50,
                   "format": "json", "timespan": f"{hours}h", "sort": "datedesc"})
        time.sleep(5.2)
        for a in (js or {}).get("articles", []):
            ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
            it = _item(a["title"], a.get("url", ""), a.get("domain", ""), ts, "gdelt")
            if c.key not in it["categories"]:
                it["categories"] = [c.key] + [x for x in it["categories"] if x != "other"]
            out.append(it)
    return out


def rss_headlines() -> list[dict]:
    import feedparser
    out = []
    for name, url in RSS_FEEDS.items():
        try:
            import httpx
            r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 RegimeRadar"})
            feed = feedparser.parse(r.content)
            for e in feed.entries[:40]:
                t = e.get("published_parsed") or e.get("updated_parsed")
                ts = (datetime(*t[:6], tzinfo=timezone.utc) if t else datetime.now(timezone.utc)).isoformat()
                out.append(_item(e.get("title", ""), e.get("link", ""), name, ts, "rss"))
        except Exception as e:
            log.warning("RSS %s failed: %s", name, e)
    return [o for o in out if o["title"]]


def _timeline(query: str, mode: str, start: datetime, end: datetime) -> pd.DataFrame:
    js = _get({"query": f"{query} sourcelang:english", "mode": mode, "format": "json",
               "STARTDATETIME": start.strftime("%Y%m%d%H%M%S"), "ENDDATETIME": end.strftime("%Y%m%d%H%M%S")})
    rows = []
    for series in (js or {}).get("timeline", []):
        for d in series.get("data", []):
            rows.append({"date": pd.to_datetime(d["date"][:8]), "value": d.get("value"), "norm": d.get("norm")})
    if not rows:
        return pd.DataFrame(columns=["value", "norm"])
    df = pd.DataFrame(rows)
    agg = {"value": "sum" if mode == "timelinevolraw" else "mean", "norm": "sum"}
    return df.groupby("date").agg(agg)


def backfill_daily(start: str = "2017-01-01", end: str | None = None, progress=print) -> pd.DataFrame:
    """Daily volume and tone per category from GDELT timelines (2017 → now).
    Incremental: only fetches days after the last stored date (minus a 7-day overlap)."""
    have = pd.read_csv(NEWS_DAILY, index_col=0, parse_dates=True) if NEWS_DAILY.exists() else None
    s = pd.Timestamp(start)
    if have is not None and len(have):
        s = max(s, have.index.max() - pd.Timedelta(days=7))
    e = pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None).normalize()
    chunks = pd.date_range(s, e + pd.Timedelta(days=1), freq="180D").tolist() + [e + pd.Timedelta(days=1)]
    cols = {}
    total = None
    for c in CATEGORIES:
        vparts, tparts = [], []
        for a, b in zip(chunks[:-1], chunks[1:]):
            progress(f"GDELT {c.label}: {a.date()} → {b.date()}")
            v = _timeline(c.gdelt_query, "timelinevolraw", a, b); time.sleep(5.2)
            t = _timeline(c.gdelt_query, "timelinetone", a, b); time.sleep(5.2)
            vparts.append(v)
            tparts.append(t)
        V = pd.concat(vparts) if vparts else pd.DataFrame()
        T = pd.concat(tparts) if tparts else pd.DataFrame()
        if len(V):
            V = V[~V.index.duplicated(keep="last")]
            cols[f"vol_{c.key}"] = V["value"]
            total = V["norm"] if total is None else total.combine(V["norm"], max)
        if len(T):
            T = T[~T.index.duplicated(keep="last")]
            cols[f"tone_{c.key}"] = T["value"]
    if not cols:
        raise RuntimeError("GDELT returned no data (network blocked or rate limited). Try again later.")
    new = pd.DataFrame(cols)
    new["vol_total"] = total
    new = new.sort_index()
    if have is not None:
        new = pd.concat([have[have.index < new.index.min()], new])
    new.to_csv(NEWS_DAILY)
    return new


# ----------------------------------------------------------------------------- demo headlines
_TEMPLATES = {
    "monetary": [("RBI holds repo rate, flags sticky inflation", -0.3), ("RBI surprises with rate cut to support growth", 0.5),
                 ("RBI governor warns on inflation risks", -0.6)],
    "fiscal": [("Fiscal deficit widens as tax collections slow", -0.5), ("Government sticks to deficit target in budget", 0.3)],
    "flows": [("FPIs pull out funds for fifth straight session", -0.7), ("Foreign investors turn net buyers of Indian equities", 0.6)],
    "regulation": [("SEBI tightens F&O rules for retail traders", -0.4), ("SEBI orders probe into trading irregularities", -0.6)],
    "global_rates": [("US Treasury yields jump on hot inflation print", -0.6), ("Fed signals pause, global yields ease", 0.5)],
    "commodities": [("Brent crude surges on supply fears", -0.5), ("Oil prices slide as OPEC output rises", 0.3)],
    "geopolitics": [("Border tensions escalate, markets on edge", -0.8), ("Ceasefire eases regional tensions", 0.6),
                    ("New tariff threat rattles Asian markets", -0.7)],
    "market_stress": [("Sensex tumbles 900 points in broad sell-off", -0.9), ("Nifty rallies to fresh record high", 0.8),
                      ("Volatility spikes as Nifty slides for third day", -0.8)],
    "corporate": [("IT majors beat Q2 estimates, guidance steady", 0.5), ("Bank earnings miss as provisions rise", -0.5)],
}


def demo_headlines(date: pd.Timestamp, row: pd.Series) -> list[dict]:
    """Deterministic simulated headlines for a date, driven by that day's simulated category surprises."""
    rng = np.random.default_rng(int(date.strftime("%Y%m%d")))
    out = []
    zs = {c.key: row.get(f"surprise_{c.key}", np.nan) for c in CATEGORIES}
    top2 = sorted([k for k, v in zs.items() if not pd.isna(v)], key=lambda k: -zs[k])[:2]
    for c in CATEGORIES:
        z = zs[c.key]
        tz = row.get(f"tonez_{c.key}", 0.0)
        if pd.isna(z) or (z < 0.5 and c.key not in top2):
            continue
        opts = _TEMPLATES[c.key]
        neg = [o for o in opts if o[1] < 0]
        pos = [o for o in opts if o[1] >= 0] or neg
        pool = neg if (tz if not np.isnan(tz) else 0) < 0 else pos
        title, tone = pool[rng.integers(len(pool))]
        ts = (date - pd.Timedelta(hours=int(rng.integers(2, 20)))).tz_localize("UTC").isoformat()
        out.append({"title": title, "url": "", "source": "Simulated wire", "published_utc": ts,
                    "categories": [c.key], "tone": tone, "origin": "simulated", "surprise": float(z)})
    return sorted(out, key=lambda x: x["published_utc"], reverse=True)


def category_snapshot(feat_row: pd.Series) -> list[dict]:
    out = []
    for c in CATEGORIES:
        z = feat_row.get(f"surprise_{c.key}", np.nan)
        tz = feat_row.get(f"tonez_{c.key}", np.nan)
        out.append({"key": c.key, "label": c.label, "color": c.color,
                    "surprise": None if pd.isna(z) else float(z), "tone_z": None if pd.isna(tz) else float(tz)})
    return out


def event_impact(df: pd.DataFrame, z_thr: float = 2.0) -> list[dict]:
    """Event study: after a category surprise (z > z_thr), how did NIFTY's next-5-day realised vol
    compare with the trailing 22-day vol, versus all other days?"""
    out = []
    ratio = (df["fwd_rv"] / df["rv22"]).replace([np.inf, -np.inf], np.nan)
    known = ratio.notna()
    for c in CATEGORIES:
        col = f"surprise_{c.key}"
        if col not in df:
            continue
        ev = (df[col] > z_thr) & known
        base = (df[col].notna()) & known & ~ev
        if ev.sum() < 5:
            out.append({"key": c.key, "label": c.label, "n_events": int(ev.sum())})
            continue
        out.append({"key": c.key, "label": c.label, "color": CAT_BY_KEY[c.key].color, "n_events": int(ev.sum()),
                    "vol_ratio_after": float(ratio[ev].median()), "vol_ratio_other": float(ratio[base].median()),
                    "uplift_pct": float(100 * (ratio[ev].median() / ratio[base].median() - 1))})
    return out
