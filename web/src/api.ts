// Data access. Talks to the FastAPI server, or — in the static GitHub Pages build —
// reads the JSON written by `python -m radar export-static`.
import type { Day, Meta, ModelInfo, Snapshot, Explain, Timeline, Track } from "./types";

export const STATIC = __STATIC__;
const STATIC_ROOT = "./static-data";

const cache = new Map<string, Promise<unknown>>();

async function getJSON<T>(url: string): Promise<T> {
  if (!cache.has(url)) {
    const p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
      return r.json();
    });
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url) as Promise<T>;
}

/** Forget a cached day so the next request fetches fresh data (live news / rebuilds). */
export function invalidateDay(date: string) {
  cache.delete(STATIC ? `${STATIC_ROOT}/days/${date}.json` : `/api/day?date=${encodeURIComponent(date)}`);
}

export const api = {
  meta: () => getJSON<Meta>(STATIC ? `${STATIC_ROOT}/meta.json` : "/api/meta"),
  timeline: () => getJSON<Timeline>(STATIC ? `${STATIC_ROOT}/timeline.json` : "/api/timeline"),
  model: () => getJSON<ModelInfo>(STATIC ? `${STATIC_ROOT}/model.json` : "/api/model"),
  track: () => getJSON<Track>(STATIC ? `${STATIC_ROOT}/track.json` : "/api/track"),
  day: (date: string) =>
    getJSON<Day>(STATIC ? `${STATIC_ROOT}/days/${date}.json` : `/api/day?date=${encodeURIComponent(date)}`),
};

export interface LiveTick {
  type: "tick" | "hello" | "news" | "rebuilt";
  source?: string;
  snapshot?: Snapshot;
  explain?: Explain;
}

/** Subscribe to live pushes (server mode only). Reconnects with back-off. */
export function subscribe(onMsg: (m: LiveTick) => void, onStatus: (ok: boolean) => void): () => void {
  if (STATIC) return () => {};
  let ws: WebSocket | null = null;
  let stop = false;
  let delay = 1000;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { delay = 1000; onStatus(true); };
    ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch { /* ignore */ } };
    ws.onclose = () => {
      onStatus(false);
      if (!stop) setTimeout(open, (delay = Math.min(delay * 2, 30000)));
    };
  };
  open();
  return () => { stop = true; ws?.close(); };
}
