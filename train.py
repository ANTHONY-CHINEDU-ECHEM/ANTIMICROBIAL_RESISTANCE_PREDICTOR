"""Train one XGBoost head per antibiotic class (multi-label resistance
prediction), sharing the same feature set across heads.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

ANTIBIOTIC_CLASSES = ["penicillins", "cephalosporins", "fluoroquinolones",
                      "carbapenems", "aminoglycosides", "glycopeptides"]
FEATURE_COLS = ["prior_abx_90d", "recent_hosp", "nursing_home", "device_catheter",
                 "device_central_line", "device_ventilator", "icu_admission",
                 "note_flag_prior_mdro", "note_flag_recent_travel", "note_flag_immunosuppressed",
                 "month"]


def main():
    df = pd.read_csv(ROOT / "data" / "culture_episodes.csv")
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["mdro_flag"])

    models = {}
    per_class_auc = {}
    for cls in ANTIBIOTIC_CLASSES:
        y_train = train_df[f"resistant_{cls}"]
        y_test = test_df[f"resistant_{cls}"]
        m = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08,
                           subsample=0.85, colsample_bytree=0.85,
                           eval_metric="auc", random_state=42, n_jobs=-1)
        m.fit(train_df[FEATURE_COLS], y_train)
        p = m.predict_proba(test_df[FEATURE_COLS])[:, 1]
        auc = roc_auc_score(y_test, p)
        per_class_auc[cls] = round(float(auc), 4)
        models[cls] = m

    joblib.dump({"models": models, "feature_cols": FEATURE_COLS, "classes": ANTIBIOTIC_CLASSES},
                MODEL_DIR / "amr_multilabel.joblib")
    test_df.to_csv(MODEL_DIR / "test_split.csv", index=False)

    metrics = {"per_class_auc_roc": per_class_auc,
               "mean_auc_roc": round(float(np.mean(list(per_class_auc.values()))), 4),
               "n_train": len(train_df), "n_test": len(test_df)}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
