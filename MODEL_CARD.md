# Model Card — Antimicrobial Resistance Predictor

## Intended use
Decision support for empiric antibiotic selection review by stewardship
pharmacists. Not for autonomous prescribing.

## Metrics (last run)
See `models/metrics.json`, `models/evaluation_report.json`, and
`models/backtest_report.json`. Typical: mean per-class AUC-ROC ≈ 0.70-0.73
(vs. 0.50 for an antibiogram-lookup-only baseline with no per-patient
discrimination); MDRO sensitivity ≈ 0.78; rolling-origin AUC drift ≈ 1-2%
across simulated retrain cycles.

## Limitations
Synthetic cohort; note-derived flags are extracted via regex here, not a
real ClinicalBERT model (see `docs/EXTENDING.md`).
