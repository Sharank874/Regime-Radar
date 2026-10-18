"""Command line entry point:  python -m radar <command>

  demo            build on the simulated market and serve the app (no internet needed)
  serve --live    real data: Yahoo Finance + GDELT/RSS, live nowcasts during NSE hours
  fetch           download / refresh market data (Yahoo Finance)
  news-backfill   download daily GDELT news timelines (2017 → today; ~15–20 min the first time)
  validate        run V0/V1/V2 and write reports/validation.md (add --live for real data)
  export-static   write a static snapshot of the app's data for GitHub Pages
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="radar", description="Regime Radar — explainable market-risk radar for NIFTY 50")
    ap.add_argument("command", choices=["demo", "serve", "fetch", "news-backfill", "validate", "export-static"])
    ap.add_argument("--live", action="store_true", help="use real market data instead of the simulator")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="web/static-public/static-data", help="export-static output folder")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "yfinance", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    mode = "live" if (a.live or a.command == "serve" and a.live) else "demo"

    if a.command in ("demo", "serve"):
        import uvicorn
        from .api import create_app
        if a.command == "demo":
            mode = "demo"
        print(f"\n  Regime Radar ({'LIVE data' if mode == 'live' else 'DEMO — simulated market'})"
              f"\n  Starting… the first start fits the models and can take 1–3 minutes."
              f"\n  Wait for the '✔ Ready' line, then open http://{a.host}:{a.port}\n")
        url = f"http://{a.host}:{a.port}"
        uvicorn.run(create_app(mode, url), host=a.host, port=a.port, log_level="warning")
    elif a.command == "fetch":
        from .data import fetch_yahoo
        d = fetch_yahoo(refresh=True)
        for k, v in d.items():
            print(f"{k:10s} {len(v):6d} rows  {v.index[0].date()} → {v.index[-1].date()}")
    elif a.command == "news-backfill":
        from .news import backfill_daily
        df = backfill_daily(progress=print)
        print(f"news_daily: {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")
    elif a.command == "validate":
        from .config import REPORT_DIR
        from .engine import Engine
        Engine(mode).build(refresh=mode == "live", validate=True)
        print((REPORT_DIR / "validation.md").read_text())
    elif a.command == "export-static":
        from .export import export_static
        export_static(mode, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
