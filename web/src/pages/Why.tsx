import { useState } from "react";
import { BeliefBars, Diverging, Waterfall } from "../charts";
import { dateLong, dateShort, pct, pts, REGIME_VAR } from "../format";
import type { Day } from "../types";

export default function Why({ day, onPick }: { day: Day; onPick: (d: string) => void }) {
  const { explain: x, snapshot: s, moves } = day;
  const ch = x.change;
  const [state, setState] = useState(2);
  const [all, setAll] = useState(false);
  const h = x.hmm;
  const notable = x.news_context.filter((c) => (c.surprise ?? 0) > 1.5).sort((a, b) => (b.surprise ?? 0) - (a.surprise ?? 0));

  return (
    <>
      <h1 className="t-xl" style={{ margin: "0 0 4px" }}>Why the risk moved on {dateLong(x.date)}</h1>
      <p className="t-sm prose" style={{ marginTop: 0 }}>
        The forecast is a logistic model, so every change can be split exactly into what each input contributed. Nothing
        here is approximated after the fact.
      </p>

      <div className="grid cols-2" style={{ marginTop: 20 }}>
        <section className="panel" aria-labelledby="h-wf">
          <h2 id="h-wf" className="t-md" style={{ margin: 0 }}>From the previous session to this one</h2>
          {ch ? (
            <>
              <p className="t-sm" style={{ margin: "2px 0 12px" }}>
                Chance of a high-volatility week went from {pct(ch.p_prev, 1)} to {pct(ch.p_now, 1)} ({pts(ch.dp)}).
                {ch.refit && " The model was refitted on a new year of data at this date, which shows up as its own step."}
              </p>
              <Waterfall start={ch.p_prev} end={ch.p_now} items={ch.items} startLabel="Previous session" endLabel="This session" />
              <p className="t-xs" style={{ marginTop: 10 }}>
                Red steps push the risk up, teal steps pull it down. Changes are exact in log-odds and shared out in
                proportion when converted to percentage points.
              </p>
            </>
          ) : <p className="muted">No previous session to compare with.</p>}
        </section>

        <section className="panel" aria-labelledby="h-level">
          <h2 id="h-level" className="t-md" style={{ margin: 0 }}>What holds the risk where it is</h2>
          <p className="t-sm" style={{ margin: "2px 0 14px" }}>
            Each input's pull on the forecast compared with an average day in the training data (log-odds units).
          </p>
          <Diverging
            rows={x.clf.contributions.slice(0, 9).map((c) => ({ label: c.label, v: c.contrib ?? 0 }))}
            unit={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`}
          />
          <p className="t-xs" style={{ marginTop: 12 }}>
            Starting point {x.clf.intercept?.toFixed(2)} plus these pulls gives log-odds {x.clf.logit?.toFixed(2)}, which is a {pct(x.clf.p, 1)} chance.
          </p>
        </section>
      </div>

      <section className="section" aria-labelledby="h-hmm">
        <h2 id="h-hmm" className="t-lg">How today's data changed the regime belief</h2>
        <p className="t-sm prose">
          The regime model updates in two moves: first it assumes regimes tend to persist (the transition step), then it
          weighs today's evidence. Each feature's share of the second move is its exact Shapley value.
        </p>
        <div className="grid cols-3">
          <div className="panel">
            <h3>Previous session's belief</h3>
            <BeliefBars names={x.state_names} values={h.prev_post} />
          </div>
          <div className="panel">
            <h3>After the persistence step</h3>
            <BeliefBars names={x.state_names} values={h.prior} />
            <p className="t-xs" style={{ marginBottom: 0 }}>Regimes rarely switch overnight, so this step only nudges the belief.</p>
          </div>
          <div className="panel">
            <h3>After today's evidence</h3>
            <BeliefBars names={x.state_names} values={h.post} />
          </div>
        </div>
        <div className="panel" style={{ marginTop: 20 }}>
          <div className="spread" style={{ flexWrap: "wrap" }}>
            <h3>Which evidence moved the belief in</h3>
            <div className="seg" role="group" aria-label="Regime">
              {x.state_names.map((n, i) => (
                <button key={n} aria-pressed={state === i} onClick={() => setState(i)}>{n}</button>
              ))}
            </div>
          </div>
          <p className="t-sm" style={{ margin: "4px 0 14px" }}>
            Probability of <b style={{ color: REGIME_VAR[state] }}>{x.state_names[state]}</b>: {pct(h.prior[state], 1)} before
            evidence, {pct(h.post[state], 1)} after.
          </p>
          <Diverging
            rows={h.features.map((f) => ({
              label: f.label,
              v: f.phi[state] ?? 0,
              note: !f.used ? <span className="faint"> (no data today)</span> : null,
            }))}
            unit={(v) => pts(v, 1)}
            max={Math.max(0.02, ...h.features.map((f) => Math.abs(f.phi[state] ?? 0)))}
          />
        </div>
      </section>

      {notable.length > 0 && !x.clf.news_in_model && (
        <section className="section" aria-labelledby="h-ctx">
          <h2 id="h-ctx" className="t-lg">Notable news, not in the model</h2>
          <p className="t-sm prose">
            Coverage in these categories was unusually heavy. News isn't a model input (it didn't pass validation), so it
            can't have caused the moves above, but it often explains them.
          </p>
          <div className="row" style={{ flexWrap: "wrap" }}>
            {notable.map((c) => (
              <span className="chip" key={c.key}><i style={{ background: c.color }} />{c.label}: {c.surprise!.toFixed(1)}σ above normal</span>
            ))}
          </div>
        </section>
      )}

      <section className="section" aria-labelledby="h-days">
        <h2 id="h-days" className="t-lg">Day by day</h2>
        <p className="t-sm">The biggest single driver of each session's change. Select a row to open that day.</p>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="data">
            <thead>
              <tr><th>Session</th><th>Regime</th><th className="n">Chance</th><th className="n">Change</th><th>Biggest driver</th></tr>
            </thead>
            <tbody>
              {(all ? moves : moves.slice(0, 12)).map((m) => (
                <tr key={m.date} className={m.date === s.date ? "hl" : ""} onClick={() => onPick(m.date)} style={{ cursor: "pointer" }}
                  tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onPick(m.date)}>
                  <td>{dateShort(m.date)}</td>
                  <td><span className="chip"><i style={{ background: REGIME_VAR[x.state_names.indexOf(m.regime)] }} />{m.regime}</span></td>
                  <td className="n">{pct(m.p, 1)}</td>
                  <td className="n" style={{ color: (m.dp ?? 0) > 0 ? "var(--up)" : "var(--down)", fontWeight: 600 }}>{pts(m.dp)}</td>
                  <td className="muted">{m.top} <span className="faint">({pts(m.top_dp)})</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {moves.length > 12 && (
          <button className="btn" style={{ marginTop: 12 }} onClick={() => setAll(!all)}>
            {all ? "Show fewer sessions" : `Show all ${moves.length} sessions`}
          </button>
        )}
      </section>
    </>
  );
}
