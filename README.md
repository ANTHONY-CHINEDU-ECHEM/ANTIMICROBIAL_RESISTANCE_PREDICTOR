# Empiric Antibiotic Resistance Prediction from Culture + EHR Data

Predicts per-antibiotic-class resistance probability at the moment of
empiric prescribing (before culture/sensitivity results return 24-72h
later), to support antibiotic stewardship review.

> **Reference-implementation note.** The briefing specifies a ClinicalBERT
> note-extraction stage feeding per-class XGBoost heads. This repo ships a
> fully-runnable version using **structured EHR + a lightweight bag-of-
> risk-flags text feature** (`src/note_flags.py`, a fast keyword/regex
> extractor) in place of a fine-tuned ClinicalBERT model, so it has no GPU
> dependency and trains in seconds on synthetic data. Swapping in a real
> fine-tuned ClinicalBERT extractor is a documented drop-in extension
> (`docs/EXTENDING.md`).

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_data.py       # synthetic culture episodes -> data/
python src/train.py                # multi-label XGBoost -> models/
python src/evaluate.py             # per-class AUC-ROC, MDRO sensitivity
python src/backtest.py             # rolling-origin drift backtest
uvicorn serving.api:app --reload
```

## Structure
| Path | Purpose |
|---|---|
| `data/generate_data.py` | Synthesizes culture episodes across 6 antibiotic classes, 8 pathogens |
| `data/culture_episodes.csv` | Generated dataset (included) |
| `src/note_flags.py` | Lightweight note-derived risk-flag extractor (ClinicalBERT stand-in) |
| `src/train.py` | Multi-label XGBoost (one head per antibiotic class) |
| `src/evaluate.py` | Per-class AUC-ROC, MDRO sensitivity, stewardship time-saved estimate |
| `src/backtest.py` | Rolling-origin (train-forward) drift backtest across 3 simulated years |
| `serving/api.py` | FastAPI `/predict` returning per-class resistance probabilities |

## License
MIT — portfolio/demonstration use. Decision-support only; always requires
antibiotic-stewardship pharmacist review before action.
