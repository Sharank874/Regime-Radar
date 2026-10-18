import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { BeliefBars, Gaussians, TransitionDiagram } from "../charts";
import { dateLong, num, pct, REGIME_VAR } from "../format";
import type { Day, ModelInfo } from "../types";

const FMT: Record<string, (v: number) => string> = {
  ret: (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}%`,
  log_rv5: (v) => `${(Math.exp(v) * 100).toFixed(0)}%`,
  log_vix: (v) => Math.exp(v).toFixed(1),
  vix_chg5: (v) => `${v >= 0 ? "+" : "−"}${Math.abs((Math.exp(v) - 1) * 100).toFixed(0)}%`,
  news_stress: (v) => `${v.toFixed(1)}σ`,
};
const HINT: Record<string, string> = {
  ret: "Today's NIFTY return",
  log_rv5: "Annualised volatility over the last 5 sessions",
  log_vix: "India VIX, the market's own volatility expectation",
  vix_chg5: "Change in India VIX over 5 sessions",
  news_stress: "How unusual and negative yesterday's news was",
};

const STEPS = [
  "Start from the previous belief",
  "Let regimes persist",
  "Weigh today's evidence",
  "Combine and normalise",
];

export default function XRay({ day }: { day: Day }) {
  const [m, setM] = useState<ModelInfo | null>(null);
  const [step, setStep] = useState<number | null>(null);
  const [auto, setAuto] = useState(false);
  const t = useRef<number | null>(null);
  useEffect(() => { api.model().then(setM); }, []);
  useEffect(() => {
    if (!auto) return;
    t.current = window.setTimeout(() => {
      if ((step ?? 0) >= 3) setAuto(false);
      else setStep((step ?? -1) + 1);
    }, 1700);
    return () => { if (t.current) clearTimeout(t.current); };
  }, [auto, step]);

  if (!m) return <div className="skeleton" style={{ height: 400 }} />;
  const x = day.explain;
  const h = x.hmm;
  const names = x.state_names;
  const from = (h.prev_post ?? []).reduce<number>((best, v, i, a) => ((v ?? 0) > (a[best] ?? 0) ? i : best), 0);
  const dim = (k: number) => step == null ? {} : ({ opacity: step === k ? 1 : 0.5, transition: "opacity 300ms ease",
    boxShadow: step === k ? "0 0 0 2px var(--model)" : undefined });

  return (
    <>
      <h1 className="t-xl" style={{ margin: "0 0 4px" }}>How the model thinks</h1>
      <p className="t-sm prose" style={{ marginTop: 0 }}>
        Two models work in sequence. A hidden Markov model keeps a running belief about which regime the market is in.
        That belief, together with volatility and India VIX features, feeds a logistic model that gives the chance of a
        high-volatility week. Here is the regime update for {dateLong(x.date)}, one step at a time.
      </p>

      <div className="row" style={{ margin: "18px 0", flexWrap: "wrap" }}>
        <button className="btn primary" onClick={() => { setStep(0); setAuto(true); }}>Walk through the update</button>
        <ol className="row" style={{ listStyle: "none", padding: 0, margin: 0, flexWrap: "wrap", gap: 6 }} aria-label="Steps">
          {STEPS.map((s, i) => (
            <li key={s}>
              <button className="btn" aria-pressed={step === i} onClick={() => { setAuto(false); setStep(i); }}
                style={step === i ? { borderColor: "var(--model)", color: "var(--model)" } : undefined}>
                {i + 1}. {s}
              </button>
            </li>
          ))}
        </ol>
      </div>

      <div className="grid cols-2">
        <section className="panel" style={dim(0)} aria-labelledby="s1">
          <h2 id="s1" className="t-md" style={{ margin: 0 }}>1. Start from the previous belief</h2>
          <p className="t-sm">What the model believed at the previous close.</p>
          <BeliefBars names={names} values={h.prev_post} />
        </section>

        <section className="panel" style={dim(1)} aria-labelledby="s2">
          <h2 id="s2" className="t-md" style={{ margin: 0 }}>2. Let regimes persist</h2>
          <p className="t-sm">
            Regimes are sticky. The transition matrix, learned from history up to {dateLong(m.train_end)}, says how
            likely each regime is to continue or switch overnight. Belief × transitions gives the starting point for today.
          </p>
          <div className="grid" style={{ gridTemplateColumns: "minmax(0,1.2fr) minmax(0,1fr)", gap: 16, alignItems: "center" }}>
            <TransitionDiagram A={m.transition} names={names} highlightFrom={step === 1 || step == null ? from : undefined} />
            <BeliefBars names={names} values={h.prior} />
          </div>
        </section>
      </div>

      <section className="panel section" style={dim(2)} aria-labelledby="s3">
        <h2 id="s3" className="t-md" style={{ margin: 0 }}>3. Weigh today's evidence</h2>
        <p className="t-sm prose">
          Each regime expects the data to look a certain way: the coloured curves. Today's value (black line) is more
          believable under some regimes than others. A value deep in the red curve's territory is evidence for Stress.
        </p>
        <div className="grid cols-2" style={{ gap: 20 }}>
          {h.features.map((f) => {
            const e = m.emissions.find((q) => q.key === f.key);
            if (!e) return null;
            const ll = f.loglik.map((v) => v ?? 0);
            const mx = Math.max(...ll);
            const w = ll.map((v) => Math.exp(v - mx));
            const tot = w.reduce((a, b) => a + b, 0);
            const best = w.indexOf(Math.max(...w));
            const raw = f.value;
            return (
              <div key={f.key}>
                <div className="spread">
                  <b>{f.label}</b>
                  <span className="t-xs">{f.used ? "" : "no data today, ignored"}</span>
                </div>
                <div className="t-xs">{HINT[f.key] ?? ""}</div>
                <Gaussians means={e.mean} stds={e.std} value={raw} names={names} label={f.label} fmt={FMT[f.key] ?? ((v) => v.toFixed(2))} />
                {f.used && (
                  <div className="row t-sm" style={{ gap: 8 }}>
                    <span>Points to</span>
                    <div style={{ display: "flex", flex: 1, height: 8, borderRadius: 3, overflow: "hidden" }} aria-hidden>
                      {w.map((v, s) => <div key={s} style={{ width: `${(v / tot) * 100}%`, background: REGIME_VAR[s] }} />)}
                    </div>
                    <b style={{ color: REGIME_VAR[best] }}>{names[best]}</b>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="panel section" style={dim(3)} aria-labelledby="s4">
        <h2 id="s4" className="t-md" style={{ margin: 0 }}>4. Combine and normalise</h2>
        <p className="t-sm prose">
          Multiply the starting belief from step 2 by how well each regime explains all of today's evidence, then rescale
          so the three add up to 100%. Evidence is down-weighted a little because these features overlap (volatility and
          VIX move together), so the model doesn't count the same signal twice.
        </p>
        <div className="grid cols-2" style={{ alignItems: "center" }}>
          <BeliefBars names={names} values={h.post} prior={h.prior} big />
          <div className="t-sm">
            <p style={{ marginTop: 0 }}>
              belief<sub>today</sub>(regime) ∝ belief<sub>step 2</sub>(regime) × Π<sub>features</sub> likelihood(value | regime)
            </p>
            <p>
              This is the forward algorithm of a hidden Markov model. It only ever uses data up to today, so the belief
              you see for any past date is exactly what the model would have said on that day.
            </p>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="h-then">
        <h2 id="h-then" className="t-lg">Then the forecast</h2>
        <p className="t-sm prose">
          The logistic model turns the regime belief and these inputs into the chance of a high-volatility week. Weights are
          per standard deviation of each input, fitted on data up to {dateLong(m.train_end)} and refitted every January.
        </p>
        <div className="grid cols-2">
          <div className="panel">
            <table className="data">
              <thead><tr><th>Input</th><th className="n">Weight</th></tr></thead>
              <tbody>
                {[...m.classifier.features].sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef)).map((f) => (
                  <tr key={f.key}>
                    <td>{f.label}</td>
                    <td className="n" style={{ color: f.coef > 0 ? "var(--up)" : "var(--down)", fontWeight: 600 }}>
                      {f.coef > 0 ? "+" : ""}{f.coef.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel">
            <h3>What each regime has looked like</h3>
            <p className="t-sm">Out-of-sample history, classifying each day by its most likely regime.</p>
            <table className="data">
              <thead><tr><th>Regime</th><th className="n">Share of days</th><th className="n">Typical spell</th><th className="n">Volatility</th><th className="n">High-vol week next</th></tr></thead>
              <tbody>
                {m.state_stats.map((r, i) => (
                  <tr key={r.name}>
                    <td><span className="chip"><i style={{ background: REGIME_VAR[i] }} />{r.name}</span></td>
                    <td className="n">{pct(r.occupancy)}</td>
                    <td className="n">{num(r.avg_run_days, 0)} days</td>
                    <td className="n">{pct(r.ann_vol)}</td>
                    <td className="n">{pct(r.stress_week_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </>
  );
}
