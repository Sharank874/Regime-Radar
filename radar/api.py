"""FastAPI server: REST views + a WebSocket that pushes live nowcast updates."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import REPORT_DIR, WEB_DIST
from .engine import Engine

log = logging.getLogger("radar.api")
IST = ZoneInfo("Asia/Kolkata")


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def send(self, msg: dict):
        data = json.dumps(msg, default=str)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def create_app(mode: str = "demo", url: str = "http://127.0.0.1:8000") -> FastAPI:
    engine = Engine(mode)
    hub = Hub()
    state = {"status": "starting", "last_live": None}

    async def live_loop():
        """Live mode: intraday nowcasts during NSE hours, news every 15 min, daily rebuild after close."""
        from . import news as N
        from .data import fetch_intraday_last
        last_news = last_daily = None
        while True:
            try:
                now = datetime.now(IST)
                weekday = now.weekday() < 5
                mins = now.hour * 60 + now.minute
                if weekday and 9 * 60 + 15 <= mins <= 15 * 60 + 40:
                    intr = await asyncio.to_thread(fetch_intraday_last)
                    snap = await asyncio.to_thread(engine.nowcast, intr)
                    if snap:
                        state["last_live"] = now.isoformat()
                        await hub.send({"type": "tick", "source": "nowcast", "snapshot": snap,
                                        "explain": engine.explain(None, provisional=True)})
                if last_news is None or (now - last_news).total_seconds() > 900:
                    st = await asyncio.to_thread(N.collect)
                    last_news = now
                    log.info("news: %s", {k: v.get("fetched") for k, v in st["sources"].items()})
                    await hub.send({"type": "news", "count": st["new_stored"]})
                if weekday and mins >= 16 * 60 and (last_daily is None or last_daily.date() < now.date()):
                    try:
                        await asyncio.to_thread(N.backfill_daily, "2017-01-01", None, log.info)
                    except Exception as e:
                        log.warning("news backfill skipped: %s", e)
                    await asyncio.to_thread(engine.build, True, False)
                    engine.provisional = None
                    last_daily = now
                    await hub.send({"type": "rebuilt", "snapshot": engine.snapshot()})
            except Exception as e:
                log.exception("live loop error: %s", e)
            await asyncio.sleep(300)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["status"] = "building"
        await asyncio.to_thread(engine.build, mode == "live", False)
        state["status"] = "ready"
        if WEB_DIST.exists():
            print(f"\n  ✔ Ready. Open {url} in your browser.\n"
                  f"    (Log lines below are background news/price updates; leave this window running.)\n", flush=True)
        else:
            print(f"\n  ✖ The web app isn't built yet, so {url} has nothing to show.\n"
                  "    Stop this with Ctrl+C, run:  cd web && npm install && npm run build && cd ..\n"
                  "    then start the server again.\n", flush=True)
        task = asyncio.create_task(live_loop()) if mode == "live" else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Regime Radar", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def ok(x):
        return JSONResponse(json.loads(json.dumps(x, default=str)))

    @app.get("/api/health")
    def health():
        return {"status": state["status"], "mode": mode, "clients": len(hub.clients)}

    @app.get("/api/meta")
    def meta():
        m = engine.meta()
        m["live_status"] = state["last_live"]
        return ok(m)

    @app.get("/api/snapshot")
    def snapshot(date: str | None = None, provisional: bool = False):
        if provisional and engine.provisional:
            return ok(engine.snapshot(None, provisional=True))
        return ok(engine.snapshot(date))

    @app.get("/api/explain")
    def explain(date: str | None = None, provisional: bool = False):
        return ok(engine.explain(date, provisional=provisional and engine.provisional is not None))

    @app.get("/api/day")
    def day(date: str | None = None):
        """Everything the UI needs for one date in a single call (used by the replay time machine)."""
        return ok({"snapshot": engine.snapshot(date), "explain": engine.explain(date), "news": engine.news(date),
                   "moves": engine.moves(30, date), "overnight": engine.overnight(date)})

    @app.get("/api/moves")
    def moves(end: str | None = None, n: int = 40):
        return ok(engine.moves(n, end))

    @app.get("/api/timeline")
    def timeline(start: str | None = None, end: str | None = None):
        return ok(engine.timeline(start, end))

    @app.get("/api/model")
    def model():
        return ok(engine.model_info())

    @app.get("/api/news")
    def news(date: str | None = None):
        return ok(engine.news(date))

    @app.get("/api/overnight")
    def overnight(date: str | None = None):
        return ok(engine.overnight(date))

    @app.get("/api/track")
    def track():
        return ok(engine.track_record())

    @app.get("/api/validation")
    def validation():
        return ok(json.loads((REPORT_DIR / "validation.json").read_text()))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        hub.clients.add(sock)
        try:
            await sock.send_text(json.dumps({"type": "hello", "mode": mode, "status": state["status"]}))
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(sock)

    if not WEB_DIST.exists():
        from fastapi.responses import HTMLResponse

        @app.get("/")
        def not_built():
            return HTMLResponse("<h1>Regime Radar server is running</h1><p>The web app isn't built yet. In the "
                                "project folder run <code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, "
                                "then restart the server.</p>")

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = WEB_DIST / path
            if path and f.is_file():
                return FileResponse(f)
            return FileResponse(WEB_DIST / "index.html")

    return app
