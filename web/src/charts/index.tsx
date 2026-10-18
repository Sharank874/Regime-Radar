import type { ReactNode } from "react";
import { pct, pts, REGIME_VAR } from "../format";
import type { Num } from "../types";

/* ---------------------------------------------------------------- belief bars */
export function BeliefBars({ names, values, prior, big }: { names: string[]; values: Num[]; prior?: Num[]; big?: boolean }) {
  return (
    <div className="stack" style={{ marginTop: 6 }}>
      {names.map((n, s) => {
        const v = values[s] ?? 0;
        const pr = prior?.[s];
        return (
          <div key={n}>
            <div className="spread" style={{ fontSize: big ? 15 : 13 }}>
              <span style={{ fontWeight: 600 }}>{n}</span>
              <span style={{ fontWeight: 700 }}>{pct(v)}</span>
            </div>
            <div style={{ position: "relative", height: big ? 12 : 8, background: "var(--sunk)", borderRadius: 3, marginTop: 4 }}>
              <div style={{ width: `${v * 100}%`, height: "100%", background: REGIME_VAR[s], borderRadius: 3,
                            transition: "width 400ms cubic-bezier(.2,.7,.2,1)" }} />
              {pr != null && (
                <div title={`before today's evidence: ${pct(pr)}`} style={{ position: "absolute", top: -3, bottom: -3, width: 2,
                  left: `calc(${pr * 100}% - 1px)`, background: "var(--ink)", opacity: 0.55 }} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- waterfall */
export interface WItem { label: string; dp: number }
export function Waterfall({ start, end, items, startLabel, endLabel }: {
  start: number; end: number; items: WItem[]; startLabel: string; endLabel: string;
}) {
  const shown = items.slice(0, 7);
  const rest = items.slice(7).reduce((a, b) => a + b.dp, 0);
  const rows: { label: string; a: number; b: number; kind: "end" | "up" | "down" }[] = [];
  rows.push({ label: startLabel, a: 0, b: start, kind: "end" });
  let acc = start;
  for (const it of shown) { rows.push({ label: it.label, a: acc, b: acc + it.dp, kind: it.dp >= 0 ? "up" : "down" }); acc += it.dp; }
  if (Math.abs(rest) > 1e-6) { rows.push({ label: "Everything else", a: acc, b: acc + rest, kind: rest >= 0 ? "up" : "down" }); acc += rest; }
  rows.push({ label: endLabel, a: 0, b: end, kind: "end" });
  const lo = Math.min(0, ...rows.map((r) => Math.min(r.a, r.b)));
  const hi = Math.max(...rows.map((r) => Math.max(r.a, r.b))) * 1.08 || 0.01;
  const W = 640, LW = 190, RW = 70, RH = 30;
  const sx = (v: number) => LW + ((v - lo) / (hi - lo)) * (W - LW - RW);
  const H = rows.length * RH + 6;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img"
      aria-label={`Stress probability moved from ${pct(start, 1)} to ${pct(end, 1)}`}>
      {rows.map((r, i) => {
        const y = i * RH + 4;
        const x0 = sx(Math.min(r.a, r.b)), x1 = sx(Math.max(r.a, r.b));
        const fill = r.kind === "end" ? "var(--model)" : r.kind === "up" ? "var(--up)" : "var(--down)";
        const val = r.kind === "end" ? pct(r.b, 1) : pts(r.b - r.a, 2);
        return (
          <g key={i}>
            <text x={LW - 10} y={y + RH / 2} textAnchor="end" dominantBaseline="middle"
              style={{ fontSize: 13, fill: r.kind === "end" ? "var(--ink)" : "var(--ink-2)", fontWeight: r.kind === "end" ? 700 : 500 }}>
              {r.label}
            </text>
            <rect x={x0} y={y + 5} width={Math.max(x1 - x0, 1.5)} height={RH - 10} rx={2} fill={fill} opacity={r.kind === "end" ? 0.9 : 1} />
            {i < rows.length - 1 && (
              <line x1={sx(r.b)} x2={sx(r.b)} y1={y + RH - 5} y2={y + RH + 5} stroke="var(--ink-3)" strokeDasharray="2 2" />
            )}
            <text x={x1 + 6} y={y + RH / 2} dominantBaseline="middle" style={{ fontSize: 12.5, fill: "var(--ink)", fontWeight: 600 }}>{val}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------------------------------------------------------------- diverging bars */
export function Diverging({ rows, unit, max }: { rows: { label: string; v: number; note?: ReactNode }[]; unit?: (v: number) => string; max?: number }) {
  const m = max ?? Math.max(...rows.map((r) => Math.abs(r.v)), 1e-6);
  return (
    <div className="stack">
      {rows.map((r) => (
        <div key={r.label} style={{ display: "grid", gridTemplateColumns: "minmax(120px, 1.1fr) 2fr 64px", gap: 10, alignItems: "center", fontSize: 13 }}>
          <span className="muted">{r.label}{r.note}</span>
          <div style={{ position: "relative", height: 10 }}>
            <div style={{ position: "absolute", left: "50%", top: -3, bottom: -3, width: 1, background: "var(--line)" }} />
            <div style={{
              position: "absolute", top: 0, height: 10, borderRadius: 2,
              background: r.v >= 0 ? "var(--up)" : "var(--down)",
              left: r.v >= 0 ? "50%" : `${50 - Math.min(1, Math.abs(r.v) / m) * 50}%`, width: `${Math.min(1, Math.abs(r.v) / m) * 50}%`,
              transition: "all 300ms ease",
            }} />
          </div>
          <span style={{ textAlign: "right", fontWeight: 600 }}>{unit ? unit(r.v) : r.v.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- gaussians (HMM emissions) */
export function Gaussians({ means, stds, value, names, label, fmt }: {
  means: number[]; stds: number[]; value: Num; names: string[]; label: string; fmt: (v: number) => string;
}) {
  const W = 360, H = 120, P = 8;
  const lo = Math.min(...means.map((m, i) => m - 3.2 * stds[i]), value ?? Infinity);
  const hi = Math.max(...means.map((m, i) => m + 3.2 * stds[i]), value ?? -Infinity);
  const sx = (v: number) => P + ((v - lo) / (hi - lo)) * (W - 2 * P);
  const dens = (v: number, m: number, s: number) => Math.exp(-0.5 * ((v - m) / s) ** 2) / s;
  const peak = Math.max(...stds.map((s) => 1 / s));
  const sy = (d: number) => H - 22 - (d / peak) * (H - 34);
  const paths = means.map((m, s) => {
    let d = "";
    for (let k = 0; k <= 120; k++) {
      const v = lo + ((hi - lo) * k) / 120;
      d += `${k ? "L" : "M"}${sx(v).toFixed(1)},${sy(dens(v, m, stds[s])).toFixed(1)}`;
    }
    return d;
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img"
      aria-label={`${label}: how each regime expects this feature to look; today's value is ${value == null ? "missing" : fmt(value)}`}>
      {paths.map((d, s) => (
        <g key={s}>
          <path d={`${d}L${sx(hi)},${H - 22}L${sx(lo)},${H - 22}Z`} fill={REGIME_VAR[s]} opacity={0.12} />
          <path d={d} fill="none" stroke={REGIME_VAR[s]} strokeWidth={2} />
        </g>
      ))}
      <line x1={P} x2={W - P} y1={H - 22} y2={H - 22} stroke="var(--line)" />
      <text x={P} y={H - 6} style={{ fontSize: 10.5, fill: "var(--ink-3)" }}>{fmt(lo)}</text>
      <text x={W - P} y={H - 6} textAnchor="end" style={{ fontSize: 10.5, fill: "var(--ink-3)" }}>{fmt(hi)}</text>
      {value != null && (
        <g>
          <line x1={sx(value)} x2={sx(value)} y1={8} y2={H - 22} stroke="var(--ink)" strokeWidth={2} />
          <text x={sx(value)} y={H - 6} textAnchor="middle" style={{ fontSize: 11, fill: "var(--ink)", fontWeight: 700 }}>{fmt(value)}</text>
        </g>
      )}
      <title>{names.join(", ")}</title>
    </svg>
  );
}

/* ---------------------------------------------------------------- calibration */
export function Calibration({ bins }: { bins: { p_mean: number; y_rate: number; n: number }[] }) {
  const W = 330, H = 300, P = 48;
  const m = Math.min(1, Math.max(0.3, ...bins.map((b) => Math.max(b.p_mean, b.y_rate))) * 1.05);
  const sx = (v: number) => P + (v / m) * (W - P - 10);
  const sy = (v: number) => H - P - (v / m) * (H - P - 10);
  const ticks = [0, m / 2, m].map((t) => Math.round(t * 10) / 10);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 380, height: "auto" }} role="img"
      aria-label="Calibration: predicted probability against how often a stress week actually followed">
      <line x1={sx(0)} y1={sy(0)} x2={sx(m)} y2={sy(m)} stroke="var(--ink-3)" strokeDasharray="4 4" />
      {ticks.map((t) => (
        <g key={t} className="axis">
          <line x1={sx(t)} x2={sx(t)} y1={sy(0)} y2={sy(0) + 4} />
          <text x={sx(t)} y={sy(0) + 16} textAnchor="middle">{pct(t)}</text>
          <line x1={sx(0) - 4} x2={sx(0)} y1={sy(t)} y2={sy(t)} />
          <text x={sx(0) - 7} y={sy(t) + 4} textAnchor="end">{pct(t)}</text>
        </g>
      ))}
      <polyline points={bins.map((b) => `${sx(b.p_mean)},${sy(b.y_rate)}`).join(" ")} fill="none" stroke="var(--model)" strokeWidth={2} />
      {bins.map((b, i) => (
        <circle key={i} cx={sx(b.p_mean)} cy={sy(b.y_rate)} r={4.5} fill="var(--surface)" stroke="var(--model)" strokeWidth={2}>
          <title>{`Predicted ${pct(b.p_mean, 1)}, happened ${pct(b.y_rate, 1)} of ${b.n} days`}</title>
        </circle>
      ))}
      <text x={(W + P) / 2} y={H - 4} textAnchor="middle" style={{ fontSize: 11, fill: "var(--ink-2)" }}>Predicted chance</text>
      <text transform={`translate(10 ${(H - P) / 2}) rotate(-90)`} textAnchor="middle" style={{ fontSize: 11, fill: "var(--ink-2)" }}>How often it happened</text>
    </svg>
  );
}

/* ---------------------------------------------------------------- heatmap */
export function Heatmap({ dates, rows }: { dates: string[]; rows: { key: string; label: string; z: Num[] }[] }) {
  const n = dates.length;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "150px 1fr", gap: "3px 10px", alignItems: "center", fontSize: 12.5 }}>
      {rows.map((r) => (
        <div key={r.key} style={{ display: "contents" }}>
          <span className="muted" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</span>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${n}, 1fr)`, gap: 1, height: 16 }}>
            {r.z.map((z, i) => {
              const v = z == null ? 0 : Math.max(-3, Math.min(3, z));
              const bg = z == null ? "var(--sunk)" : v > 0
                ? `color-mix(in srgb, var(--stress) ${Math.round((v / 3) * 85)}%, var(--sunk))`
                : `color-mix(in srgb, var(--calm) ${Math.round((-v / 3) * 45)}%, var(--sunk))`;
              return <div key={i} title={`${dates[i]}: ${z == null ? "no data" : `${z > 0 ? "+" : ""}${z.toFixed(1)}σ`}`} style={{ background: bg, borderRadius: 1 }} />;
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- transition diagram */
export function TransitionDiagram({ A, names, highlightFrom }: { A: number[][]; names: string[]; highlightFrom?: number }) {
  const W = 420, H = 300;
  const pos = [[80, 220], [210, 70], [340, 220]];
  const R = 46;
  const arrows: ReactNode[] = [];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      if (i === j) continue;
      const [x1, y1] = pos[i], [x2, y2] = pos[j];
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
      const nx = -dy / L, ny = dx / L, off = 12;
      const sx = x1 + (dx / L) * R + nx * off, sy = y1 + (dy / L) * R + ny * off;
      const ex = x2 - (dx / L) * (R + 6) + nx * off, ey = y2 - (dy / L) * (R + 6) + ny * off;
      const w = 1 + Math.sqrt(A[i][j]) * 14;
      const on = highlightFrom == null || highlightFrom === i;
      arrows.push(
        <g key={`${i}${j}`} opacity={on ? 1 : 0.25}>
          <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="var(--ink-3)" strokeWidth={w} markerEnd="url(#arr)" />
          <text x={(sx + ex) / 2 + nx * 14} y={(sy + ey) / 2 + ny * 14} textAnchor="middle" dominantBaseline="middle"
            style={{ fontSize: 11.5, fill: "var(--ink-2)", fontWeight: 600 }}>{pct(A[i][j], 1)}</text>
        </g>,
      );
    }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 460, height: "auto" }} role="img"
      aria-label="Transition probabilities between regimes from one day to the next">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
          <path d="M0,0L10,5L0,10z" fill="var(--ink-3)" />
        </marker>
      </defs>
      {arrows}
      {pos.map(([x, y], s) => (
        <g key={s} opacity={highlightFrom == null || highlightFrom === s ? 1 : 0.6}>
          <circle cx={x} cy={y} r={R} fill="var(--surface)" stroke={REGIME_VAR[s]} strokeWidth={3} />
          <text x={x} y={y - 6} textAnchor="middle" style={{ fontSize: 14, fontWeight: 700, fill: "var(--ink)" }}>{names[s]}</text>
          <text x={x} y={y + 13} textAnchor="middle" style={{ fontSize: 11.5, fill: "var(--ink-2)" }}>stays {pct(A[s][s], 1)}</text>
        </g>
      ))}
    </svg>
  );
}

/* ---------------------------------------------------------------- small line chart */
export function MiniLines({ series, labels, height = 160, fmt }: {
  series: { name: string; color: string; values: Num[]; dash?: string }[]; labels: string[]; height?: number; fmt: (v: number) => string;
}) {
  const W = 600, H = height, P = 34;
  const all = series.flatMap((s) => s.values.filter((v): v is number => v != null));
  const lo = Math.min(...all) * 0.95, hi = Math.max(...all) * 1.05;
  const sx = (i: number) => P + (i / Math.max(labels.length - 1, 1)) * (W - P - 10);
  const sy = (v: number) => H - 22 - ((v - lo) / (hi - lo || 1)) * (H - 34);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Score by year">
      {[lo, (lo + hi) / 2, hi].map((t, k) => (
        <g key={k} className="axis">
          <line x1={P} x2={W - 10} y1={sy(t)} y2={sy(t)} strokeDasharray="2 4" />
          <text x={P - 6} y={sy(t) + 4} textAnchor="end">{fmt(t)}</text>
        </g>
      ))}
      {labels.map((l, i) => (
        <text key={l} x={sx(i)} y={H - 6} textAnchor="middle" style={{ fontSize: 10.5, fill: "var(--ink-3)" }}>{l}</text>
      ))}
      {series.map((s) => (
        <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash}
          points={s.values.map((v, i) => (v == null ? "" : `${sx(i)},${sy(v)}`)).join(" ")} />
      ))}
    </svg>
  );
}
