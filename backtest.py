"""Rolling-origin backtest: train on months [0, m), test on the next 6
months, sliding forward across the simulated 36-month history — validates
how much AUC degrades from local antibiogram drift between retrains.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).parent.parent
from train import ANTIBIOTIC_CLASSES, FEATURE_COLS  # noqa: E402
import sys
sys.path.insert(0, str(Path(__file__).parent))


def main():
    df = pd.read_csv(ROOT / "data" / "culture_episodes.csv")
    results = []
    for origin in [12, 18, 24]:
        train_df = df[df["month"] < origin]
        test_df = df[(df["month"] >= origin) & (df["month"] < origin + 6)]
        if len(test_df) < 100:
            continue
        fold_aucs = []
        for cls in ANTIBIOTIC_CLASSES:
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                               random_state=42, n_jobs=-1, eval_metric="auc")
            m.fit(train_df[FEATURE_COLS], train_df[f"resistant_{cls}"])
            p = m.predict_proba(test_df[FEATURE_COLS])[:, 1]
            fold_aucs.append(roc_auc_score(test_df[f"resistant_{cls}"], p))
        results.append({"train_origin_month": origin, "test_window": f"{origin}-{origin+6}",
                         "mean_auc_roc": round(float(np.mean(fold_aucs)), 4), "n_test": len(test_df)})

    static_auc = results[0]["mean_auc_roc"] if results else None
    latest_auc = results[-1]["mean_auc_roc"] if results else None
    drift_pct = None
    if static_auc and latest_auc:
        drift_pct = round((static_auc - latest_auc) / static_auc * 100, 2)

    report = {"rolling_origin_folds": results, "auc_drift_pct_first_to_last": drift_pct,
              "meets_5pct_retention_target": bool(drift_pct is not None and abs(drift_pct) <= 5.0)}
    with open(ROOT / "models" / "backtest_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
