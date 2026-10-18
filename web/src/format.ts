import type { Num } from "./types";

export const pct = (x: Num | undefined, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
export const pts = (x: Num | undefined, d = 1) =>
  x == null ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(d)} pts`;
export const signedPct = (x: Num | undefined, d = 2) =>
  x == null ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(d)}%`;
export const num = (x: Num | undefined, d = 0) =>
  x == null ? "–" : x.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const dateLong = (s: string) => {
  const [y, m, d] = s.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
};
export const dateShort = (s: string) => {
  const [, m, d] = s.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]}`;
};
export const monthYear = (s: string) => {
  const [y, m] = s.split("-").map(Number);
  return `${MONTHS[m - 1]} ${y}`;
};

export const REGIME_VAR = ["var(--calm)", "var(--choppy)", "var(--stress)"];
export const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

export const LEVEL_TEXT: Record<string, string> = {
  Low: "Low",
  Moderate: "Moderate",
  Elevated: "Elevated",
  High: "High",
};

import type { NewsView } from "./types";

/** Explains an empty headline list instead of just saying "no news". */
export function emptyNewsText(n: NewsView): string {
  const c = n.collection;
  if (!c) return "No headlines stored for this date. Article lists only reach back about three months.";
  const src = Object.entries(c.sources).map(([k, v]) => `${k.toUpperCase()} ${v.error ? "failed" : `${v.fetched} found`}`).join(", ");
  const when = new Date(c.last_run_utc).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
  if (c.total_stored === 0) return `No headlines collected yet. Last check at ${when}: ${src}. The server checks every 15 minutes.`;
  return `No headlines in this date's window. ${c.total_stored} stored so far; last check at ${when} (${src}). Article lists only reach back about three months.`;
}

/** "led by short-term volatility" style phrase from a change decomposition */
export function driverPhrase(items: { label: string; dp: number }[], dp: number): string {
  const same = items.filter((i) => Math.sign(i.dp) === Math.sign(dp) && Math.abs(i.dp) > 0.0005);
  if (!same.length) return "";
  const PROPER = new Set(["India", "S&P", "NIFTY", "HMM", "VIX", "Fed", "RBI", "SEBI"]);
  const lc = (s: string) => (/^[A-Z][a-z]/.test(s) && !PROPER.has(s.split(" ")[0]) ? s[0].toLowerCase() + s.slice(1) : s);
  const names = same.slice(0, 2).map((i) => lc(i.label));
  return names.length === 1 ? `mostly from ${names[0]}` : `mostly from ${names[0]} and ${names[1]}`;
}
