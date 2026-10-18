import { useEffect, useState } from "react";
import { api } from "../api";
import { Calibration, MiniLines } from "../charts";
import { dateLong, dateShort, num, pct } from "../format";
import type { LedgerRow, Meta, Track } from "../types";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const STATUS_TEXT: Record<string, string> = {
  pass: "Passed", partial: "Partly passed", weak: "Inconclusive", fail: "Failed", insufficient: "Not run yet",
};

export default function TrackRecord({ meta }: { meta: Meta }) {
  const [t, setT] = useState<Track | null>(null);
  const [tab, setTab] = useState<"live" | "backtest">("live");
  useEffect(() => { api.track().then(setT); }, []);
  if (!t) return <div className="skeleton" style={{ height: 400 }} />;
  const v = t.validation;
  const v0 = v.v0, v1 = v.v1, v2 = v.v2;
  const order = ["p_model", "p_vix", "p_har", "p_garch", "p_gbm", "p_clim"];
  const rows: LedgerRow[] = tab === "live" ? (t.live.rows ?? []) : (t.backtest.recent ?? []);

  return (
    <>
      <h1 className="t-xl" style={{ margin: "0 0 4px" }}>Track record</h1>
      <p className="t-sm prose" style={{ marginTop: 0 }}>
        Every score here is out of sample: each year was forecast by models fitted only on earlier years, with a 5-day gap
        so no training target overlaps the test year. {meta.mode === "demo" && <b>These are simulated-market results; they prove the pipeline, not the idea.</b>}
      </p>

      <div className="grid cols-3" style={{ marginTop: 20 }}>
        <section className="panel" aria-labelledby="v0">
          <div className="spread"><h2 id="v0" className="t-md" style={{ margin: 0 }}>Audit of the original project</h2></div>
          <p className="t-sm">Why 98% accuracy on K-Means labels isn't a forecast.</p>
          <dl style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "6px 12px", fontSize: 14, margin: 0 }}>
            <dt className="muted">Random forest, shuffled CV, same-day labels</dt><dd style={{ margin: 0, fontWeight: 700 }}>{pct(v0.shuffled_cv_accuracy.random_forest, 1)}</dd>
            <dt className="muted">SVM, shuffled CV, same-day labels</dt><dd style={{ margin: 0, fontWeight: 700 }}>{pct(v0.shuffled_cv_accuracy.svm, 1)}</dd>
            <dt className="muted">Random forest, label {v0.future_label_chronological.horizon_days} days ahead</dt><dd style={{ margin: 0, fontWeight: 700 }}>{pct(v0.future_label_chronological.random_forest_balanced_accuracy, 1)}</dd>
            <dt className="muted">"Nothing changes" rule</dt><dd style={{ margin: 0, fontWeight: 700 }}>{pct(v0.future_label_chronological.persistence_balanced_accuracy, 1)}</dd>
          </dl>
          <p className="t-xs" style={{ marginBottom: 0 }}>Forward-looking figures are balanced accuracy on a chronological split.</p>
        </section>

        <section className="panel" aria-labelledby="v1">
          <div className="spread"><h2 id="v1" className="t-md" style={{ margin: 0 }}>Does it beat simple models?</h2>
            <span className={`verdict ${v1.status}`}>{STATUS_TEXT[v1.status]}</span></div>
          <p className="t-sm">{cap(v1.verdict.replace(/^[A-Z ()]+— /, ""))}.</p>
          <p style={{ fontSize: 14, margin: 0 }}>
            Brier score versus {v1.names[v1.best_baseline]}: <b>{v1.model_minus_best_brier.diff > 0 ? "+" : ""}{v1.model_minus_best_brier.diff.toFixed(4)}</b>{" "}
            <span className="muted">(90% range {v1.model_minus_best_brier.ci90[0].toFixed(4)} to {v1.model_minus_best_brier.ci90[1].toFixed(4)}; below zero means the model is better)</span>
          </p>
        </section>

        <section className="panel" aria-labelledby="v2">
          <div className="spread"><h2 id="v2" className="t-md" style={{ margin: 0 }}>Does news help?</h2>
            <span className={`verdict ${v2.status}`}>{STATUS_TEXT[v2.status]}</span></div>
          <p className="t-sm">{cap(v2.verdict.replace(/^[A-Z ()]+— /, ""))}</p>
          {v2.news_minus_price_brier && (
            <p style={{ fontSize: 14, margin: 0 }}>
              Brier with news minus without: <b>{v2.news_minus_price_brier.diff > 0 ? "+" : ""}{v2.news_minus_price_brier.diff.toFixed(4)}</b>{" "}
              <span className="muted">(90% range {v2.news_minus_price_brier.ci90[0].toFixed(4)} to {v2.news_minus_price_brier.ci90[1].toFixed(4)})</span>
            </p>
          )}
          <p className="t-xs" style={{ marginBottom: 0 }}>News enters the live model only if this test passes. Currently: {v.use_news_in_production ? "on" : "off"}.</p>
        </section>
      </div>

      <section className="section" aria-labelledby="h-cmp">
        <h2 id="h-cmp" className="t-lg">Head to head, {dateLong(v1.period[0])} to {dateLong(v1.period[1])}</h2>
        <p className="t-sm prose">
          Target: {v.target}. Lower Brier is better; Brier skill is the improvement over always predicting the usual rate.
        </p>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="data">
            <thead>
              <tr><th>Model</th><th className="n">PR-AUC</th><th className="n">ROC-AUC</th><th className="n">Brier</th><th className="n">Brier skill</th></tr>
            </thead>
            <tbody>
              {order.filter((k) => v1.table[k]?.brier != null).map((k) => {
                const s = v1.table[k];
                return (
                  <tr key={k} className={k === "p_model" ? "hl" : ""}>
                    <td style={{ fontWeight: k === "p_model" ? 700 : 500 }}>{v1.names[k]}</td>
                    <td className="n">{s.pr_auc?.toFixed(3)}</td>
                    <td className="n">{s.roc_auc?.toFixed(3)}</td>
                    <td className="n">{s.brier?.toFixed(4)}</td>
                    <td className="n">{s.brier_skill != null ? `${(s.brier_skill * 100).toFixed(1)}%` : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid cols-2 section">
        <section className="panel" aria-labelledby="h-cal">
          <h2 id="h-cal" className="t-md" style={{ margin: 0 }}>Does 30% mean 30%?</h2>
          <p className="t-sm">Days grouped by forecast. Points on the dashed line mean the stated chances came true as often as claimed.</p>
          <Calibration bins={v1.calibration} />
        </section>
        <section className="panel" aria-labelledby="h-year">
          <h2 id="h-year" className="t-md" style={{ margin: 0 }}>Year by year</h2>
          <p className="t-sm">Brier score per test year (lower is better). Stress years are hard for every model.</p>
          <MiniLines
            labels={v1.by_year.map((y) => `’${String(y.year).slice(2)}`)}
            fmt={(x) => x.toFixed(2)}
            height={240}
            series={[
              { name: "Regime Radar", color: "var(--model)", values: v1.by_year.map((y) => y.p_model) },
              { name: "India VIX only", color: "var(--choppy)", values: v1.by_year.map((y) => y.p_vix) },
              { name: "HAR-RV", color: "var(--ink-3)", values: v1.by_year.map((y) => y.p_har), dash: "4 4" },
            ]}
          />
          <div className="legend" style={{ marginTop: 6 }}>
            <span><i className="swatch" style={{ background: "var(--model)" }} />Regime Radar</span>
            <span><i className="swatch" style={{ background: "var(--choppy)" }} />India VIX only</span>
            <span><i className="swatch" style={{ background: "var(--ink-3)" }} />HAR-RV</span>
          </div>
        </section>
      </div>

      <section className="section" aria-labelledby="h-led">
        <div className="spread" style={{ flexWrap: "wrap" }}>
          <h2 id="h-led" className="t-lg" style={{ margin: 0 }}>Prediction ledger</h2>
          <div className="seg" role="group" aria-label="Ledger">
            <button aria-pressed={tab === "live"} onClick={() => setTab("live")}>Live ({num(t.live.n)})</button>
            <button aria-pressed={tab === "backtest"} onClick={() => setTab("backtest")}>Back-test, latest 30</button>
          </div>
        </div>
        <p className="t-sm prose">
          Live forecasts are written once, timestamped, before the outcome is known, and scored automatically five
          sessions later. They are never mixed with back-test results.
        </p>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="data">
            <thead><tr><th>Session</th><th>Recorded</th><th className="n">Forecast</th><th className="n">Range</th><th>Regime</th><th>Outcome</th></tr></thead>
            <tbody>
              {[...rows].reverse().map((r) => (
                <tr key={r.asof + r.kind}>
                  <td>{dateShort(r.asof)} {r.asof.slice(0, 4)}</td>
                  <td className="muted">{r.created_utc ? new Date(r.created_utc).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) : "walk-forward"}</td>
                  <td className="n" style={{ fontWeight: 700 }}>{pct(r.p, 1)}</td>
                  <td className="n muted">{pct(r.p_lo)}–{pct(r.p_hi)}</td>
                  <td>{r.regime}</td>
                  <td>{r.outcome == null ? <span className="faint">Pending</span>
                    : r.outcome ? <b style={{ color: "var(--stress)" }}>High-vol week</b> : <span style={{ color: "var(--calm)" }}>Normal week</span>}</td>
                </tr>
              ))}
              {!rows.length && (
                <tr><td colSpan={6} className="muted">
                  {meta.mode === "demo"
                    ? "Live forecasts are recorded only with real data. Start the server with --live and the first entry appears after the next close."
                    : "No live forecasts yet. The first one is recorded after the next close."}
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
