import { Diverging, Heatmap } from "../charts";
import { dateLong, dateShort, emptyNewsText, num } from "../format";
import type { Day } from "../types";

function when(iso: string) {
  const d = new Date(iso);
  return d.toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Kolkata" });
}

export default function News({ day }: { day: Day }) {
  const n = day.news;
  const cats = Object.fromEntries(n.categories.map((c) => [c.key, c]));
  const impact = [...n.impact].filter((r) => r.uplift_pct != null).sort((a, b) => (b.uplift_pct ?? 0) - (a.uplift_pct ?? 0));

  return (
    <>
      <h1 className="t-xl" style={{ margin: "0 0 4px" }}>News around {dateLong(n.date)}</h1>
      <p className="t-sm prose" style={{ marginTop: 0 }}>
        {n.use_news_in_model
          ? "News is a model input: unusual coverage in these categories feeds the forecast."
          : `News is shown for context. It is not a model input because adding it didn't improve forecasts out of sample (validation test V2). ${num(n.news_days)} days of category history are available.`}
      </p>

      <div className="grid cols-hero" style={{ marginTop: 20 }}>
        <section className="panel" aria-labelledby="h-tape" style={{ padding: "16px 0 4px" }}>
          <h2 id="h-tape" className="t-md" style={{ margin: "0 20px 8px" }}>Headlines</h2>
          <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
            {n.headlines.map((h) => (
              <li key={h.published_utc + h.title} style={{ padding: "11px 20px", borderTop: "1px solid var(--line)",
                display: "grid", gridTemplateColumns: "10px 1fr", gap: 12 }}>
                <span aria-label={h.tone < -0.15 ? "negative tone" : h.tone > 0.15 ? "positive tone" : "neutral tone"}
                  style={{ width: 10, height: 10, marginTop: 6, borderRadius: 2,
                           background: h.tone < -0.15 ? "var(--stress)" : h.tone > 0.15 ? "var(--calm)" : "var(--ink-3)" }} />
                <div>
                  <div style={{ fontWeight: 600 }}>
                    {h.url ? <a href={h.url} target="_blank" rel="noreferrer" style={{ color: "inherit" }}>{h.title}</a> : h.title}
                  </div>
                  <div className="row t-xs" style={{ gap: 8, marginTop: 4, flexWrap: "wrap" }}>
                    <span>{h.source}</span>
                    <span>{when(h.published_utc)} IST</span>
                    {h.categories.filter((c) => cats[c]).map((c) => (
                      <span className="chip" key={c}><i style={{ background: cats[c].color }} />{cats[c].label}</span>
                    ))}
                  </div>
                </div>
              </li>
            ))}
            {!n.headlines.length && (
              <li className="muted" style={{ padding: "12px 20px", borderTop: "1px solid var(--line)" }}>
                {emptyNewsText(n)} The coverage chart uses daily category counts, which go back to 2017 once you run
                <code> python -m radar news-backfill</code>.
              </li>
            )}
          </ul>
        </section>

        <section className="panel" aria-labelledby="h-cov">
          <h2 id="h-cov" className="t-md" style={{ margin: 0 }}>Coverage compared with normal</h2>
          <p className="t-sm" style={{ margin: "2px 0 14px" }}>
            How unusual each category's share of news was the day before, in standard deviations from its last 60 days.
          </p>
          <Diverging
            rows={n.categories.map((c) => ({ label: c.label, v: c.surprise ?? 0 }))}
            unit={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}σ`}
            max={3}
          />
        </section>
      </div>

      <section className="section" aria-labelledby="h-heat">
        <h2 id="h-heat" className="t-lg">The last 60 sessions</h2>
        <p className="t-sm">Darker red means coverage far above normal; teal means quieter than usual.</p>
        <div className="panel">
          <Heatmap dates={n.heatmap.dates} rows={n.heatmap.rows} />
          <div className="spread t-xs" style={{ marginTop: 6, paddingLeft: 160 }}>
            <span>{n.heatmap.dates[0] && dateShort(n.heatmap.dates[0])}</span>
            <span>{n.heatmap.dates.length > 0 && dateShort(n.heatmap.dates[n.heatmap.dates.length - 1])}</span>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="h-imp">
        <h2 id="h-imp" className="t-lg">What followed big news surprises before</h2>
        <p className="t-sm prose">
          Event study over the full history: on days after a category ran more than 2σ above normal, how NIFTY's next-week
          realised volatility compared with its previous month, against all other days. This is descriptive, not a
          forecast.
        </p>
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="data">
            <thead>
              <tr><th>Category</th><th className="n">Surprise days</th><th className="n">Next-week vol vs last month</th><th className="n">On other days</th><th className="n">Difference</th></tr>
            </thead>
            <tbody>
              {impact.map((r) => (
                <tr key={r.key}>
                  <td><span className="chip"><i style={{ background: r.color }} />{r.label}</span></td>
                  <td className="n">{r.n_events}</td>
                  <td className="n">{r.vol_ratio_after?.toFixed(2)}×</td>
                  <td className="n">{r.vol_ratio_other?.toFixed(2)}×</td>
                  <td className="n" style={{ fontWeight: 700, color: (r.uplift_pct ?? 0) > 0 ? "var(--up)" : "var(--down)" }}>
                    {(r.uplift_pct ?? 0) > 0 ? "+" : ""}{r.uplift_pct?.toFixed(0)}%
                  </td>
                </tr>
              ))}
              {!impact.length && <tr><td colSpan={5} className="muted">Not enough news history yet. Run <code>python -m radar news-backfill</code>.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
