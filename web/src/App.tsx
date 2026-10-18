import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, invalidateDay, STATIC, subscribe } from "./api";
import Ribbon from "./components/Ribbon";
import { dateLong } from "./format";
import Radar from "./pages/Radar";
import Why from "./pages/Why";
import News from "./pages/News";
import XRay from "./pages/XRay";
import TrackRecord from "./pages/Track";
import type { Day, Meta, Timeline } from "./types";

const PAGES = [
  { id: "radar", label: "Radar" },
  { id: "why", label: "Why it moved" },
  { id: "news", label: "News" },
  { id: "model", label: "How the model thinks" },
  { id: "track", label: "Track record" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

const SPANS = [
  { label: "6M", n: 126 },
  { label: "1Y", n: 252 },
  { label: "3Y", n: 756 },
  { label: "All", n: 99999 },
];

function readHash(): { page: PageId; date: string | null } {
  const [p, d] = location.hash.replace(/^#\/?/, "").split("/");
  const page = (PAGES.find((x) => x.id === p)?.id ?? "radar") as PageId;
  return { page, date: d && /^\d{4}-\d{2}-\d{2}$/.test(d) ? d : null };
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [tl, setTl] = useState<Timeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [{ page, date: hashDate }, setRoute] = useState(readHash);
  const [date, setDate] = useState<string | null>(null);
  const [day, setDay] = useState<Day | null>(null);
  const [span, setSpan] = useState(252);
  const [live, setLive] = useState<{ connected: boolean; tick: Day | null }>({ connected: false, tick: null });
  const [replay, setReplay] = useState<{ ep: number; playing: boolean; speed: number } | null>(null);

  // initial load
  useEffect(() => {
    Promise.all([api.meta(), api.timeline()])
      .then(([m, t]) => { setMeta(m); setTl(t); })
      .catch((e) => setError(String(e)));
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const allowed = useMemo(() => (meta?.static_dates ? new Set(meta.static_dates) : null), [meta]);
  const latest = tl ? tl.dates[tl.dates.length - 1] : null;
  const current = date ?? hashDate ?? latest;
  const following = !date && !hashDate;

  // keep the URL shareable: #/page/date
  const go = useCallback((p: PageId, d?: string | null) => {
    const dd = d === undefined ? (date ?? hashDate) : d;
    history.replaceState(null, "", `#/${p}${dd ? `/${dd}` : ""}`);
    setRoute({ page: p, date: dd ?? null });
  }, [date, hashDate]);

  const pick = useCallback((d: string) => {
    setDate(d === latest ? null : d);
    history.replaceState(null, "", `#/${page}${d === latest ? "" : `/${d}`}`);
    if (d === latest) setRoute({ page, date: null });
  }, [latest, page]);

  // fetch the selected day (with a small prefetch for smooth replay)
  useEffect(() => {
    if (!current) return;
    let alive = true;
    api.day(current).then((d) => alive && setDay(d)).catch((e) => alive && setError(String(e)));
    return () => { alive = false; };
  }, [current]);

  // live pushes (server mode)
  useEffect(() => subscribe(
    (m) => {
      if (m.type === "tick" && m.snapshot && m.explain && day) {
        setLive((s) => ({ ...s, tick: { ...day, snapshot: m.snapshot!, explain: m.explain! } }));
      }
      if ((m.type === "news" || m.type === "rebuilt") && current) {
        invalidateDay(current);
        api.day(current).then(setDay).catch(() => {});
      }
    },
    (ok) => setLive((s) => ({ ...s, connected: ok })),
  ), [day, current]);

  // replay time machine
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!replay?.playing || !meta || !tl) return;
    const ep = meta.episodes[replay.ep];
    const dates = tl.dates.filter((d) => d >= ep.start && d <= ep.end && (!allowed || allowed.has(d)));
    let i = Math.max(0, dates.indexOf(current ?? ""));
    if (i >= dates.length - 1) i = 0;
    const tick = () => {
      const d = dates[i];
      if (!d) { setReplay((r) => (r ? { ...r, playing: false } : r)); return; }
      api.day(d).then(() => pick(d));
      if (dates[i + 1]) api.day(dates[i + 1]);
      i += 1;
      timer.current = window.setTimeout(tick, replay.speed);
    };
    tick();
    return () => { if (timer.current) clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replay?.playing, replay?.ep, replay?.speed, meta, tl]);

  if (error) return <Failure msg={error} />;
  if (!meta || !tl || !current) return <Loading />;

  const shown = following && live.tick ? live.tick : day;
  const provisional = !!shown?.snapshot.provisional;
  const statusDot = replay?.playing ? "replay" : following && live.connected && meta.mode === "live" ? "live" : "";
  const statusText = replay?.playing
    ? "Replaying"
    : provisional
      ? `Provisional, from a delayed ${new Date(shown!.snapshot.as_of_print!).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })} price`
      : following
        ? meta.mode === "live" ? "Latest close" : "Latest simulated close"
        : "Viewing history";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <Mark /> Regime Radar <small>NIFTY 50</small>
        </div>
        <nav className="nav" aria-label="Sections">
          {PAGES.map((p) => (
            <button key={p.id} aria-current={p.id === page ? "page" : undefined} onClick={() => go(p.id)}>{p.label}</button>
          ))}
        </nav>
        <div className="status">
          <span className={`dot ${statusDot}`} />
          <span className="long">{statusText}</span>
          <b>{dateLong(shown?.snapshot.date ?? current)}</b>
        </div>
      </header>

      {meta.mode === "demo" && (
        <div className="banner" role="note">
          Demo mode: this is a simulated market, so prices, news and scores are synthetic. Run{" "}
          <code>python -m radar serve --live</code> for real NIFTY 50 data.
        </div>
      )}

      <section className="ribbon-wrap" aria-label="Timeline">
        <div className="ribbon-head">
          <div className="legend">
            <span><i className="swatch" style={{ background: "var(--model)", height: 3 }} /> Chance of a high-volatility week</span>
            <span><i className="swatch" style={{ background: "var(--calm)" }} /> Calm</span>
            <span><i className="swatch" style={{ background: "var(--choppy)" }} /> Choppy</span>
            <span><i className="swatch" style={{ background: "var(--stress)" }} /> Stress</span>
            <span><i className="swatch" style={{ background: "var(--stress)", width: 2 }} /> A high-vol week followed</span>
          </div>
          <div className="ribbon-controls">
            <div className="seg" role="group" aria-label="Zoom">
              {SPANS.map((s) => (
                <button key={s.label} aria-pressed={span === s.n} onClick={() => setSpan(s.n)}>{s.label}</button>
              ))}
            </div>
            <select className="select" aria-label="Replay a stress episode" value={replay?.ep ?? ""}
              onChange={(e) => {
                const ep = Number(e.target.value);
                if (Number.isNaN(ep) || e.target.value === "") return;
                pick(meta.episodes[ep].start);
                setSpan(126);
                setReplay({ ep, playing: true, speed: replay?.speed ?? 450 });
              }}>
              <option value="">Replay a stress episode…</option>
              {meta.episodes.map((e, i) => (
                <option key={e.peak} value={i}>Around {dateLong(e.peak)}</option>
              ))}
            </select>
            {replay && (
              <>
                <button className="btn" onClick={() => setReplay({ ...replay, playing: !replay.playing })}>
                  {replay.playing ? "Pause" : "Play"}
                </button>
                <select className="select" aria-label="Replay speed" value={replay.speed}
                  onChange={(e) => setReplay({ ...replay, speed: Number(e.target.value) })}>
                  <option value={900}>Slow</option>
                  <option value={450}>Normal</option>
                  <option value={160}>Fast</option>
                </select>
              </>
            )}
            {!following && (
              <button className="btn" onClick={() => { setReplay(null); setDate(null); history.replaceState(null, "", `#/${page}`); setRoute({ page, date: null }); }}>
                Back to latest
              </button>
            )}
          </div>
        </div>
        <Ribbon tl={tl} date={shown?.snapshot.date && tl.dates.includes(shown.snapshot.date) ? shown.snapshot.date : current}
          onDate={(d) => { if (replay?.playing) setReplay({ ...replay, playing: false }); pick(d); }}
          span={span} compact={page !== "radar"} allowed={allowed} />
      </section>

      <main className="page" id="main">
        {!shown ? <Loading inline /> : (
          <>
            {page === "radar" && <Radar day={shown} meta={meta} onNav={(p) => go(p as PageId)} />}
            {page === "why" && <Why day={shown} onPick={pick} />}
            {page === "news" && <News day={shown} />}
            {page === "model" && <XRay day={shown} />}
            {page === "track" && <TrackRecord meta={meta} />}
          </>
        )}
      </main>
      <footer className="page t-xs" style={{ paddingTop: 0 }}>
        Regime Radar {meta.version}. {meta.provenance.source}. Not investment advice: it estimates risk, it does not
        predict direction or recommend trades.{STATIC ? " Static snapshot: live updates need the Python server." : ""}
      </footer>
    </div>
  );
}

function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
      <rect x="3" y="8" width="6" height="18" rx="1" fill="var(--calm)" />
      <rect x="13" y="4" width="6" height="22" rx="1" fill="var(--choppy)" />
      <rect x="23" y="12" width="6" height="14" rx="1" fill="var(--stress)" />
    </svg>
  );
}

function Loading({ inline }: { inline?: boolean }) {
  return (
    <div className={inline ? "" : "page"} aria-busy="true">
      <div className="skeleton" style={{ height: 120, marginBottom: 20 }} />
      <div className="skeleton" style={{ height: 260 }} />
      <p className="t-sm">Loading the model…</p>
    </div>
  );
}

function Failure({ msg }: { msg: string }) {
  return (
    <div className="page prose">
      <h1 className="t-xl">Can't reach the Regime Radar server</h1>
      <p>The app couldn't load its data ({msg}).</p>
      <p>Start the server with <code>python -m radar demo</code> (simulated market) or{" "}
        <code>python -m radar serve --live</code> (real data), then reload this page. The first start takes about a
        minute while the models are fitted.</p>
    </div>
  );
}
