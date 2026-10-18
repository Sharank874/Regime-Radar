"""Export a static copy of the app's data so the UI can be hosted on GitHub Pages
(no Python server). Live nowcasts and the news feed refresh are server-only features."""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT
from .engine import Engine


def _w(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=str, separators=(",", ":")))


def export_static(mode: str, out: str, recent_days: int = 260):
    e = Engine(mode).build()
    o = Path(out)
    if not o.is_absolute():
        o = ROOT / o
    meta = e.meta()
    meta["static"] = True
    dates = e.dates()
    keep = set(dates[-recent_days:])
    for ep in meta["episodes"]:
        keep |= {d for d in dates if ep["start"] <= d <= ep["end"]}
    meta["static_dates"] = sorted(keep)
    _w(o / "meta.json", meta)
    _w(o / "timeline.json", e.timeline())
    _w(o / "model.json", e.model_info())
    _w(o / "track.json", e.track_record())
    _w(o / "overnight.json", e.overnight())
    for d in sorted(keep):
        _w(o / "days" / f"{d}.json", {"snapshot": e.snapshot(d), "explain": e.explain(d),
                                      "news": e.news(d), "moves": e.moves(30, d),
                                      "overnight": e.overnight(d)})
    print(f"exported {len(keep)} days to {o}")
