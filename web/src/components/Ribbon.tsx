import { useEffect, useMemo, useRef, useState } from "react";
import type { Timeline } from "../types";
import { dateLong, monthYear, pct } from "../format";

interface Props {
  tl: Timeline;
  date: string;
  onDate: (d: string) => void;
  span: number; // trading days visible
  compact?: boolean;
  allowed?: Set<string> | null; // static build: only exported dates can be selected
}

const W = 1200;

/** The belief ribbon: every trading day as a sliver coloured by the model's regime
 *  belief, with the stress probability line above it. Doubles as the date scrubber. */
export default function Ribbon({ tl, date, onDate, span, compact, allowed }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const dragging = useRef(false);
  const [px, setPx] = useState(1200);
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setPx(e.contentRect.width || 1200));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const n = tl.dates.length;
  const sel = Math.max(0, tl.dates.indexOf(date));

  const [i0, i1] = useMemo(() => {
    const s = Math.min(span, n);
    let end = n - 1;
    let start = end - s + 1;
    if (sel < start) {
      start = Math.max(0, sel - Math.floor(s / 2));
      end = Math.min(n - 1, start + s - 1);
    }
    return [start, end];
  }, [span, n, sel]);

  const H_LINE = compact ? 54 : 104;
  const H_BELIEF = compact ? 22 : 40;
  const GAP = 6;
  const H_AX = 18;
  const H = H_LINE + GAP + H_BELIEF + 8 + H_AX;
  const cnt = i1 - i0 + 1;
  const x = (i: number) => ((i - i0) / Math.max(cnt - 1, 1)) * W;
  const dx = W / Math.max(cnt - 1, 1);

  const pmax = useMemo(() => {
    let m = 0.5;
    for (let i = i0; i <= i1; i++) m = Math.max(m, tl.hi[i] ?? tl.p[i] ?? 0);
    return Math.min(1, Math.ceil(m * 10) / 10);
  }, [tl, i0, i1]);
  const yP = (p: number) => H_LINE - (p / pmax) * (H_LINE - 4);

  const paths = useMemo(() => {
    let line = "", band = "", bandLow = "";
    const layers = ["", "", ""];
    const top = H_LINE + GAP;
    for (let i = i0; i <= i1; i++) {
      const xi = x(i);
      const p = tl.p[i];
      if (p != null) line += `${line ? "L" : "M"}${xi.toFixed(1)},${yP(p).toFixed(1)}`;
      const hi = tl.hi[i], lo = tl.lo[i];
      if (hi != null && lo != null) {
        band += `${band ? "L" : "M"}${xi.toFixed(1)},${yP(hi).toFixed(1)}`;
        bandLow = `L${xi.toFixed(1)},${yP(lo).toFixed(1)}` + bandLow;
      }
      // stacked belief: calm at the bottom, stress at the top
      let acc = 0;
      for (let s = 0; s < 3; s++) {
        const v = tl.post[s][i] ?? 0;
        const y0 = top + H_BELIEF * (1 - acc);
        const y1 = top + H_BELIEF * (1 - acc - v);
        layers[s] += `M${(xi - dx / 2).toFixed(1)},${y0.toFixed(1)}H${(xi + dx / 2 + 0.6).toFixed(1)}V${y1.toFixed(1)}H${(xi - dx / 2).toFixed(1)}Z`;
        acc += v;
      }
    }
    return { line, band: band ? band + bandLow + "Z" : "", layers };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tl, i0, i1, pmax, compact]);

  const ticks = useMemo(() => {
    const out: { i: number; label: string }[] = [];
    const step = cnt > 1500 ? 12 : cnt > 600 ? 3 : 1; // months between labels
    let last = "";
    for (let i = i0; i <= i1; i++) {
      const ym = tl.dates[i].slice(0, 7);
      if (ym !== last) {
        last = ym;
        const m = Number(ym.slice(5, 7));
        if ((m - 1) % step === 0) out.push({ i, label: step >= 12 ? ym.slice(0, 4) : monthYear(tl.dates[i]).replace(" 20", " ’") });
      }
    }
    // drop labels that would collide (closer than ~5% of the width)
    const minGap = (64 / px) * (cnt - 1);
    const kept: typeof out = [];
    for (const t of out) {
      if (t.i - i0 < minGap * 0.3) continue;
      if (!kept.length || t.i - kept[kept.length - 1].i >= minGap) kept.push(t);
    }
    return kept;
  }, [tl, i0, i1, cnt, px]);

  const outcomes = useMemo(() => {
    let d = "";
    for (let i = i0; i <= i1; i++) if (tl.y[i] === 1) d += `M${x(i).toFixed(1)},${H - H_AX - 6}v5`;
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tl, i0, i1, H]);

  const idxAt = (clientX: number) => {
    const r = svgRef.current!.getBoundingClientRect();
    const f = (clientX - r.left) / r.width;
    let i = Math.round(i0 + f * (cnt - 1));
    i = Math.max(i0, Math.min(i1, i));
    if (allowed && !allowed.has(tl.dates[i])) {
      // snap to nearest exported date
      for (let k = 1; k < n; k++) {
        if (i - k >= 0 && allowed.has(tl.dates[i - k])) return i - k;
        if (i + k < n && allowed.has(tl.dates[i + k])) return i + k;
      }
    }
    return i;
  };

  const base = tl.base_rate[sel] ?? tl.base_rate[n - 1] ?? 0.2;
  const hv = hover ?? sel;
  const tipX = (x(hv) / W) * 100;

  const step = (k: number) => {
    let i = sel + k;
    while (i >= 0 && i < n && allowed && !allowed.has(tl.dates[i])) i += Math.sign(k);
    if (i >= 0 && i < n) onDate(tl.dates[i]);
  };

  return (
    <div className="ribbon" style={{ position: "relative" }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ height: H }}
        role="slider"
        tabIndex={0}
        aria-label="Date. Use arrow keys to move one trading day."
        aria-valuemin={0}
        aria-valuemax={n - 1}
        aria-valuenow={sel}
        aria-valuetext={dateLong(date)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
          if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
          if (e.key === "PageUp") { step(-20); e.preventDefault(); }
          if (e.key === "PageDown") { step(20); e.preventDefault(); }
          if (e.key === "End") { onDate(tl.dates[n - 1]); e.preventDefault(); }
        }}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          onDate(tl.dates[idxAt(e.clientX)]);
        }}
        onPointerMove={(e) => {
          const i = idxAt(e.clientX);
          setHover(i);
          if (dragging.current) onDate(tl.dates[i]);
        }}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => { setHover(null); dragging.current = false; }}
      >
        <line x1={0} x2={W} y1={yP(base)} y2={yP(base)} stroke="var(--ink-3)" strokeDasharray="3 4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        <path d={paths.band} fill="var(--model-soft)" />
        <path d={paths.line} fill="none" stroke="var(--model)" strokeWidth={1.6} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        {paths.layers.map((d, s) => (
          <path key={s} d={d} fill={["var(--calm)", "var(--choppy)", "var(--stress)"][s]} />
        ))}
        <path d={outcomes} stroke="var(--stress)" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
        {hover != null && hover !== sel && (
          <line x1={x(hover)} x2={x(hover)} y1={0} y2={H - H_AX} stroke="var(--ink-3)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        )}
        <line x1={x(sel)} x2={x(sel)} y1={0} y2={H - H_AX + 2} stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      </svg>
      <div aria-hidden style={{ position: "absolute", left: 0, right: 0, top: H - H_AX + 2, height: H_AX }}>
        {ticks.map((t) => (
          <span key={t.i} style={{ position: "absolute", left: `${(x(t.i) / W) * 100}%`, fontSize: 11,
            color: "var(--ink-3)", whiteSpace: "nowrap",
            transform: x(t.i) / W > 0.92 ? "translateX(-100%)" : "translateX(2px)" }}>{t.label}</span>
        ))}
      </div>
      {/* text overlays live outside the stretched SVG so letters don't distort */}
      <div
        aria-hidden
        style={{
          position: "absolute", top: 0, left: `clamp(0px, calc(${tipX}% + 8px), calc(100% - 190px))`,
          fontSize: 12, background: "var(--surface)", padding: "2px 6px", borderRadius: 4,
          boxShadow: "0 0 0 1px var(--line)", pointerEvents: "none", whiteSpace: "nowrap",
        }}
      >
        <b>{dateLong(tl.dates[hv])}</b>&nbsp; stress {pct(tl.p[hv])}
      </div>
      <div aria-hidden style={{ position: "absolute", right: 0, top: yP(base) * 1 - 16, fontSize: 11, color: "var(--ink-3)" }}>
        usual {pct(base)}
      </div>
    </div>
  );
}
