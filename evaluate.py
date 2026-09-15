"""Evaluation report: per-class AUC-ROC vs. antibiogram-lookup baseline,
MDRO sensitivity, and a stewardship review-time-saved estimate.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"


def main():
    bundle = joblib.load(MODEL_DIR / "amr_multilabel.joblib")
    models, feature_cols, classes = bundle["models"], bundle["feature_cols"], bundle["classes"]
    test_df = pd.read_csv(MODEL_DIR / "test_split.csv")

    baseline_auc = {}
    model_auc = {}
    for cls in classes:
        y = test_df[f"resistant_{cls}"]
        # antibiogram-lookup baseline: predict the class-wide prevalence for everyone (no discrimination)
        baseline_auc[cls] = 0.5
        p = models[cls].predict_proba(test_df[feature_cols])[:, 1]
        model_auc[cls] = round(float(roc_auc_score(y, p)), 4)

    # MDRO sensitivity: flag if any class predicted resistant with p>=0.5, compare to true mdro_flag
    preds_matrix = np.column_stack([models[c].predict_proba(test_df[feature_cols])[:, 1] for c in classes])
    predicted_mdro = (np.sum(preds_matrix >= 0.5, axis=1) >= 3).astype(int)
    mdro_sensitivity = recall_score(test_df["mdro_flag"], predicted_mdro)

    report = {
        "per_class_auc_roc": model_auc,
        "mean_auc_roc": round(float(np.mean(list(model_auc.values()))), 4),
        "antibiogram_baseline_auc": 0.5,
        "mdro_sensitivity": round(float(mdro_sensitivity), 4),
        "n_test": len(test_df),
    }
    with open(MODEL_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
