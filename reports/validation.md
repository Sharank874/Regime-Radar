# Regime Radar — validation report

Generated 2026-10-18T21:02:58+00:00 · source: Yahoo Finance via yfinance; news via GDELT DOC 2.0 + RSS

**Target:** 1 if annualised realised vol over the next 5 sessions exceeds the 80th percentile of the training window.

## V0 — Audit of the original design

K-Means (k=3) cluster sizes: [1780, 254, 2599]

| Test | Accuracy |
|---|---:|
| Random Forest, shuffled 10-fold CV on same-day K-Means labels | 97.7% |
| SVM, shuffled 5-fold CV on same-day K-Means labels | 99.0% |
| Random Forest, label 5 days ahead, chronological split | 78.7% (balanced 77.5%) |
| Naive persistence (label stays the same) | 78.5% (balanced 78.1%) |

Shuffled cross-validation on K-Means labels mainly measures how well a classifier can redraw the K-Means boundaries from the same features, so very high accuracy is expected and is not evidence of forecasting. Predicting the label in the future with a chronological split, and comparing with the naive 'regime stays the same' rule, is the meaningful test.

## V1 — Price-only model vs baselines (walk-forward, out-of-sample)

Period 2014-01-02 → 2026-10-16

| Model | PR-AUC | ROC-AUC | Brier | Brier skill |
|---|---:|---:|---:|---:|
| Regime Radar (HMM + logistic) | 0.525 | 0.868 | 0.0495 | 0.291 |
| Gradient boosting (same features) | 0.528 | 0.853 | 0.0495 | 0.291 |
| India VIX only | 0.459 | 0.855 | 0.0534 | 0.235 |
| HAR-RV | 0.402 | 0.830 | 0.0584 | 0.163 |
| GARCH(1,1) | 0.410 | 0.827 | 0.0586 | 0.161 |
| Base rate (climatology) | 0.073 | 0.468 | 0.0853 | -0.222 |

Model − best baseline (India VIX only) Brier: -0.0039 (90% block-bootstrap CI -0.0075 to -0.0011; negative = model better)

**Verdict: PASS — beats every baseline including the best one**

## V2 — Does news add skill?

**Verdict: NOT RUN — not enough news history. Run `python -m radar news-backfill` (GDELT timelines go back to 2017) and re-run validation.**

## Production decision

News features in the production model: **OFF** (switched on only if V2 passes).
