import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import ANTIBIOTIC_CLASSES, FEATURE_COLS

DATA_PATH = ROOT / "data" / "culture_episodes.csv"
MODEL_PATH = ROOT / "models" / "amr_multilabel.joblib"


def test_data_exists():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 1000
    for cls in ANTIBIOTIC_CLASSES:
        assert f"resistant_{cls}" in df.columns
        assert 0.05 < df[f"resistant_{cls}"].mean() < 0.95


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first")
def test_model_predicts_all_classes():
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).head(20)
    for cls in ANTIBIOTIC_CLASSES:
        p = bundle["models"][cls].predict_proba(df[FEATURE_COLS])[:, 1]
        assert (p >= 0).all() and (p <= 1).all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first")
def test_beats_random_baseline():
    from sklearn.metrics import roc_auc_score
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    for cls in ANTIBIOTIC_CLASSES:
        p = bundle["models"][cls].predict_proba(df[FEATURE_COLS])[:, 1]
        auc = roc_auc_score(df[f"resistant_{cls}"], p)
        assert auc > 0.6, f"{cls} AUC {auc:.3f} too low"
