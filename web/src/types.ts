export type Num = number | null;

export interface Meta {
  mode: "demo" | "live";
  version: string;
  built_utc: string;
  first_date: string;
  last_date: string;
  provenance: { source: string; series?: Record<string, unknown> };
  use_news: boolean;
  news_days: number;
  state_names: string[];
  horizon: number;
  v1: string;
  v2: string;
  episodes: { peak: string; start: string; end: string; rv22: number }[];
  static?: boolean;
  static_dates?: string[];
  live_status?: string | null;
}

export interface Snapshot {
  date: string;
  provisional: boolean;
  mode: string;
  as_of_print: string | null;
  market: { close: Num; ret: Num; rv5: Num; rv22: Num; vix: Num; dd252: Num; spx_ret_lag: Num };
  regime: { names: string[]; post: Num[]; prior: Num[]; label: string; confidence: Num };
  stress: {
    p: Num; lo: Num; hi: Num; base_rate: Num; level: string | null; ratio: Num; prev: Num;
    threshold_vol: Num; horizon: number;
  };
  baselines: { vix: Num; har: Num; garch: Num; gbm: Num };
  outcome: { y: Num; fwd_rv: Num };
  fold: number;
}

export interface HmmFeature {
  key: string; label: string; value: Num; z: Num; phi: Num[]; loglik: Num[]; used: boolean;
}

export interface ChangeItem { key: string; label: string; dlogit: number; dp: number }

export interface Explain {
  date: string;
  state_names: string[];
  hmm: {
    prev_post: Num[]; prior: Num[]; post: Num[]; drift: Num[] | null;
    features: HmmFeature[]; news_in_model: boolean;
  };
  clf: {
    intercept: Num; logit: Num; p: Num; news_in_model: boolean;
    contributions: { key: string; label: string; value: Num; contrib: Num }[];
  };
  news_context: Category[];
  change: { p_prev: number; p_now: number; dp: number; dlogit: number; refit: boolean; items: ChangeItem[] } | null;
}

export interface Category { key: string; label: string; color: string; surprise: Num; tone_z: Num }

export interface Headline {
  title: string; url: string; source: string; published_utc: string;
  categories: string[]; tone: number; origin: string; surprise?: number;
}

export interface NewsView {
  date: string;
  headlines: Headline[];
  categories: Category[];
  news_stress: Num;
  impact: { key: string; label: string; color?: string; n_events: number; vol_ratio_after?: number;
            vol_ratio_other?: number; uplift_pct?: number }[];
  heatmap: { dates: string[]; rows: { key: string; label: string; color: string; z: Num[] }[] };
  use_news_in_model: boolean;
  news_days: number;
  collection?: { last_run_utc: string; new_stored: number; total_stored: number;
                 sources: Record<string, { fetched: number; error?: string }> } | null;
}

export interface Move { date: string; p: Num; dp: Num; top: string | null; top_dp: Num; regime: string }

export interface Overnight {
  spx_session: string; spx_ret: number; spx_pctile: number; spx_abs_pctile: number; us_vix_chg: Num;
  bucket: number; bucket_label: string;
  conditional: { bucket: string; n: number; p_big_move?: number; p_down?: number; median_ret?: number; median_gap?: Num };
  base: { p_big_move: number; p_down: number };
  table: Overnight["conditional"][];
  gap_reliable: boolean;
  nifty_last_close: string;
}

export interface Day { snapshot: Snapshot; explain: Explain; news: NewsView; moves: Move[]; overnight: Overnight }

export interface Timeline {
  dates: string[]; close: Num[]; vix: Num[]; post: Num[][]; p: Num[]; lo: Num[]; hi: Num[];
  p_vix: Num[]; y: Num[]; fwd_rv: Num[]; rv5: Num[]; regime: number[]; base_rate: Num[];
}

export interface Emission {
  key: string; label: string; mean: number[]; std: number[]; mean_z: number[]; std_z: number[];
  center: number; scale: number;
}

export interface ModelInfo {
  version: string; fold_year: number; train_end: string; state_names: string[];
  transition: number[][]; emissions: Emission[];
  state_stats: { name: string; occupancy: Num; avg_run_days: Num; expected_duration: Num;
                 mean_daily_ret: Num; ann_vol: Num; stress_week_rate: Num }[];
  classifier: { features: { key: string; label: string; coef: number }[]; intercept: number };
  folds: Record<string, unknown>[];
  config: { horizon: number; stress_quantile: number; n_states: number; hmm_features: string[]; sticky: number };
  use_news: boolean;
}

export interface Score {
  n: number; base_rate?: number; pr_auc?: number; roc_auc?: number; brier?: number;
  log_loss?: number; brier_skill?: number;
}

export interface Validation {
  generated_utc: string; mode: string; version: string;
  provenance: { source: string };
  target: string;
  v0: {
    features: string[]; cluster_sizes: number[];
    shuffled_cv_accuracy: { random_forest: number; svm: number };
    future_label_chronological: { horizon_days: number; random_forest_accuracy: number;
      random_forest_balanced_accuracy: number; persistence_accuracy: number; persistence_balanced_accuracy: number };
    verdict: string;
  };
  v1: {
    table: Record<string, Score>; names: Record<string, string>; best_baseline: string;
    model_minus_best_brier: { diff: number; ci90: [number, number] };
    model_minus_har_brier: { diff: number; ci90: [number, number] };
    status: string; verdict: string; period: [string, string];
    by_year: { year: number; p_model: Num; p_vix: Num; p_har: Num; p_clim: Num; stress_rate: number }[];
    calibration: { p_mean: number; y_rate: number; n: number }[];
  };
  v2: {
    status: string; verdict: string; news_days: number; period?: [string, string];
    with_news?: Score; price_only?: Score;
    news_minus_price_brier?: { diff: number; ci90: [number, number] };
  };
  use_news_in_production: boolean;
}

export interface Track {
  backtest: Score & { calibration?: { p_mean: number; y_rate: number; n: number }[]; recent?: LedgerRow[] };
  live: Score & { scored?: number; rows?: LedgerRow[] };
  validation: Validation;
}

export interface LedgerRow {
  asof: string; kind: string; created_utc: string | null; p: number; p_lo: number; p_hi: number;
  threshold: number; model_version: string; regime: string; outcome: number | null;
  realized_vol: number | null; scored_utc: string | null;
}
