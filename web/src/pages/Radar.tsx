import { BeliefBars } from "../charts";
import { dateLong, dateShort, driverPhrase, emptyNewsText, num, pct, REGIME_VAR, signedPct } from "../format";
import type { Day, Meta } from "../types";

const LEVEL_COLOR: Record<string, string> = {
  Low: "var(--calm)", Moderate: "var(--ink-2)", Elevated: "var(--choppy)", High: "var(--stress)",
};

export default function Radar({ day, meta, onNav }: { day: Day; meta: Meta; onNav: (p: string) => void }) {
  const { snapshot: s, explain: x, news, overnight: o } = day;
  const st = s.stress;
  const ch = x.change;
  const k = s.regime.names.indexOf(s.regime.label);
  const max = Math.max(0.6, (st.hi ?? 0) * 1.15);
  const pos = (v: number | null) => `${((v ?? 0) / max) * 100}%`;

  return (
    <>
      <div className="grid cols-hero">
        {/* hero: the forecast */}
        <section aria-labelledby="h-chance">
          <h1 id="h-chance" className="t-md" style={{ margin: 0 }}>Chance of a high-volatility week</h1>
          <p className="t-sm" style={{ margin: "2px 0 14px" }}>
            NIFTY 50, next {st.horizon} sessions after the close of {dateLong(s.date)}
          </p>
          <div className="row" style={{ alignItems: "flex-end", gap: 20, flexWrap: "wrap" }}>
            <div className="t-hero" aria-live="polite">{pct(st.p)}</div>
            <div style={{ paddingBottom: 10 }}>
              <div className="t-lg" style={{ color: LEVEL_COLOR[st.level ?? "Moderate"] }}>{st.level}</div>
              <div className="muted">
                {(st.ratio ?? 0) < 0.25 ? `well below the usual ${pct(st.base_rate)}` : `${num(st.ratio, 1)} times the usual ${pct(st.base_rate)}`}
              </div>
            </div>
          </div>

          {/* uncertainty strip */}
          <div style={{ position: "relative", height: 34, marginTop: 18, maxWidth: 560 }} aria-hidden>
            <div style={{ position: "absolute", top: 12, left: 0, right: 0, height: 6, background: "var(--sunk)", borderRadius: 3 }} />
            <div style={{ position: "absolute", top: 12, height: 6, left: pos(st.lo), width: `calc(${pos(st.hi)} - ${pos(st.lo)})`,
                          background: "var(--model-soft)", boxShadow: "inset 0 0 0 1px var(--model)", borderRadius: 3, transition: "all 350ms ease" }} />
            <div style={{ position: "absolute", top: 6, height: 18, width: 3, left: `calc(${pos(st.p)} - 1.5px)`, background: "var(--model)",
                          borderRadius: 2, transition: "left 350ms ease" }} />
            <div style={{ position: "absolute", top: 4, height: 22, width: 1, left: pos(st.base_rate), background: "var(--ink-3)" }} />
            <span className="t-xs" style={{ position: "absolute", top: 26, left: pos(st.base_rate), transform: "translateX(-50%)" }}>usual</span>
          </div>
          <p className="t-sm" style={{ marginTop: 6, maxWidth: 560 }}>
            The shaded range ({pct(st.lo)} to {pct(st.hi)}) is where 80% of the model's bootstrap re-fits land, so treat
            the headline number as an estimate, not a precise reading.
          </p>

          {ch && (
            <p className="t-md" style={{ fontWeight: 500, maxWidth: 560 }}>
              {Math.abs(ch.dp) < 0.0005
                ? "Unchanged from the previous session."
                : `${ch.dp > 0 ? "Up" : "Down"} ${(Math.abs(ch.dp) * 100).toFixed(1)} points since the previous session, ${driverPhrase(ch.items, ch.dp) || "with no single dominant driver"}.`}{" "}
              <button className="btn" style={{ marginLeft: 4 }} onClick={() => onNav("why")}>See the full breakdown</button>
            </p>
          )}

          {s.outcome.y != null && (
            <div className="panel" style={{ marginTop: 18, maxWidth: 560, borderColor: s.outcome.y ? "var(--stress)" : "var(--line)" }}>
              <h3>What actually happened</h3>
              <p style={{ margin: 0 }}>
                Over the next {st.horizon} sessions NIFTY's realised volatility was <b>{pct(s.outcome.fwd_rv)}</b> a year,
                against a high-volatility line of {pct(st.threshold_vol)}.{" "}
                {s.outcome.y ? <b style={{ color: "var(--stress)" }}>A high-volatility week did follow.</b>
                  : <b style={{ color: "var(--calm)" }}>It stayed below the line.</b>}
              </p>
            </div>
          )}
        </section>

        {/* regime belief */}
        <section aria-labelledby="h-regime" className="panel">
          <h2 id="h-regime" className="t-md" style={{ margin: 0 }}>Which regime is the market in?</h2>
          <p className="t-sm" style={{ margin: "2px 0 10px" }}>
            Hidden Markov model belief after today's evidence. The thin mark shows the belief before today's data arrived.
          </p>
          <BeliefBars names={s.regime.names} values={s.regime.post} prior={s.regime.prior} big />
          <p style={{ marginTop: 16, marginBottom: 12 }}>
            The model is <b>{pct(s.regime.confidence)}</b> sure the market is in{" "}
            <b style={{ color: REGIME_VAR[k] }}>{s.regime.label}</b>.
          </p>
          <button className="btn" onClick={() => onNav("model")}>See how it decided</button>
        </section>
      </div>

      <div className="grid cols-3 section">
        <section className="panel" aria-labelledby="h-mkt">
          <h3 id="h-mkt">Market at the close</h3>
          <dl style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "8px 12px", margin: "10px 0 0", fontSize: 14 }}>
            <dt className="muted">NIFTY 50</dt>
            <dd style={{ margin: 0, fontWeight: 700 }}>
              {num(s.market.close, 2)}{" "}
              <span style={{ color: (s.market.ret ?? 0) < 0 ? "var(--stress)" : "var(--calm)" }}>{signedPct(s.market.ret)}</span>
            </dd>
            <dt className="muted">India VIX</dt><dd style={{ margin: 0, fontWeight: 600 }}>{num(s.market.vix, 2)}</dd>
            <dt className="muted">5-day realised volatility</dt><dd style={{ margin: 0, fontWeight: 600 }}>{pct(s.market.rv5, 1)}</dd>
            <dt className="muted">20-day realised volatility</dt><dd style={{ margin: 0, fontWeight: 600 }}>{pct(s.market.rv22, 1)}</dd>
            <dt className="muted">Below 1-year high</dt><dd style={{ margin: 0, fontWeight: 600 }}>{pct(-(s.market.dd252 ?? 0), 1)}</dd>
          </dl>
        </section>

        <section className="panel" aria-labelledby="h-ovn">
          <h3 id="h-ovn">Overnight from Wall Street</h3>
          <p style={{ margin: "10px 0 8px" }}>
            The S&amp;P 500 {o.spx_ret >= 0 ? "rose" : "fell"} <b>{pct(Math.abs(o.spx_ret), 2)}</b> on {dateShort(o.spx_session)},
            a bigger move than {pct(o.spx_abs_pctile)} of sessions.
          </p>
          {o.conditional.p_big_move != null && (
            <p className="muted" style={{ margin: 0, fontSize: 14 }}>
              After sessions like that ({o.conditional.bucket.toLowerCase().replace("s&p ", "")}), NIFTY's next day moved more than 1%{" "}
              <b style={{ color: "var(--ink)" }}>{pct(o.conditional.p_big_move)}</b> of the time, against {pct(o.base.p_big_move)} on an
              average day ({num(o.conditional.n)} past cases).
            </p>
          )}
        </section>

        <section className="panel" aria-labelledby="h-news">
          <div className="spread">
            <h3 id="h-news">In the news</h3>
            <button className="btn" onClick={() => onNav("news")}>All news</button>
          </div>
          <ul style={{ listStyle: "none", padding: 0, margin: "8px 0 0" }}>
            {news.headlines.slice(0, 4).map((h) => (
              <li key={h.published_utc + h.title} style={{ padding: "7px 0", borderBottom: "1px solid var(--line)", fontSize: 14 }}>
                <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, marginRight: 8,
                               background: h.tone < -0.15 ? "var(--stress)" : h.tone > 0.15 ? "var(--calm)" : "var(--ink-3)" }} />
                {h.title}
              </li>
            ))}
            {!news.headlines.length && <li className="muted" style={{ fontSize: 14 }}>{emptyNewsText(news)}</li>}
          </ul>
          <p className="t-xs" style={{ margin: "8px 0 0" }}>
            {news.use_news_in_model ? "News is one of the model's inputs." : "Shown as context: news is not a model input because it didn't pass validation test V2."}
          </p>
        </section>
      </div>

      <section className="section" aria-labelledby="h-second">
        <h2 id="h-second" className="t-lg">Second opinions</h2>
        <p className="t-sm">The same question answered by simpler models. If they disagree a lot, the extra machinery is doing something; if not, it isn't adding much today.</p>
        <div className="panel" style={{ maxWidth: 760 }}>
          {[
            ["Regime Radar", st.p, "var(--model)"],
            ["India VIX only", s.baselines.vix, "var(--ink-2)"],
            ["HAR-RV (past volatility)", s.baselines.har, "var(--ink-2)"],
            ["GARCH(1,1)", s.baselines.garch, "var(--ink-2)"],
            ["Usual rate", st.base_rate, "var(--ink-3)"],
          ].map(([label, v, c]) => (
            <div key={label as string} style={{ display: "grid", gridTemplateColumns: "190px 1fr 52px", gap: 12, alignItems: "center", padding: "5px 0", fontSize: 14 }}>
              <span className={label === "Regime Radar" ? "" : "muted"} style={{ fontWeight: label === "Regime Radar" ? 700 : 500 }}>{label as string}</span>
              <div style={{ height: 8, background: "var(--sunk)", borderRadius: 3 }}>
                <div style={{ height: 8, width: `${Math.min(1, (v as number) ?? 0) * 100}%`, background: c as string, borderRadius: 3, transition: "width 350ms ease" }} />
              </div>
              <span style={{ textAlign: "right", fontWeight: 600 }}>{pct(v as number)}</span>
            </div>
          ))}
          <p className="t-xs" style={{ margin: "10px 0 0" }}>
            Back-test verdict for this model: {meta.v1 === "pass" ? "beats the simple baselines" : meta.v1 === "partial" ? "beats past-volatility models but not India VIX on its own" : "does not beat past-volatility models"}.{" "}
            <button className="btn" onClick={() => onNav("track")} style={{ marginLeft: 4 }}>Track record</button>
          </p>
        </div>
      </section>
    </>
  );
}
