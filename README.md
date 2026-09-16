# 🛡️ AMR Predictor — Antimicrobial Resistance Risk & Stewardship System

**A production-grade, multi-label machine learning system that predicts patient-specific antibiotic resistance probabilities at the point of care — before culture results exist.**

[![Status](https://img.shields.io/badge/status-reference%20implementation-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Model](https://img.shields.io/badge/model-XGBoost%20%7C%20LightGBM-orange)]()
[![Validation](https://img.shields.io/badge/validation-temporal%20rolling--origin-critical)]()

> **Built by [Anthony Chinedu Echem](#-author)** — Machine Learning Engineer & Data Scientist
> *End-to-end ML portfolio project — data design, modeling, temporal validation & deployment.*

<img width="1307" height="816" alt="AMR Predictor — Interactive Stewardship Dashboard" src="https://github.com/user-attachments/assets/794ac2e8-5db7-4317-8e57-5122603ca020" />

<p align="center"><em>The reference dashboard: headline KPIs, per-class discrimination, rolling-origin backtest stability, cohort composition, and a permanent model-card summary — all in one pharmacist-facing view.</em></p>

---

## 📌 Executive Summary

Empiric antibiotic therapy has to be chosen **before** laboratory culture and susceptibility results exist — a 48–72 hour blind spot in which clinicians rely on judgment, quarterly hospital antibiograms, and patient history alone. Choosing too broad a spectrum accelerates resistance across the hospital network; choosing too narrow a spectrum risks treatment failure. **AMR Predictor closes that gap** with a per-patient, per-antibiotic-class resistance probability generated at the exact moment a culture is ordered, using only data already sitting in the chart.

This repository is a **reference implementation**: a synthetic-but-realistic 18,000-episode cohort, a fully reproducible modeling pipeline, rigorous temporal (not just random) validation, and a pharmacist-facing dashboard — packaged the way a real stewardship-program deliverable would be, including its limitations stated up front rather than buried.

---

## 🎯 Quantified Impact — What the Numbers Mean in Practice

> **Methodology note:** All figures below are computed on a held-out or rolling-origin test split of the project's **synthetic** 18,000-episode cohort (see [Data & Disclosure](#-data-transparency--disclosure)). They are reported here as **modeled / illustrative operational impact** — the kind of business case a stewardship program would build *before* a pilot — not as claims of realized clinical outcomes. Every number is benchmarked against an honest, no-information baseline, never an inflated straw man.

| Metric | Result | What it means operationally |
|---|---|---|
| **Mean AUC-ROC** | **0.717** across 6 resistance classes | **+43.4% discrimination lift** over the 0.50 blind-guess floor — the model reliably rank-orders which patients are more likely to be resistant, turning a coin-flip into a triageable risk signal |
| **MDRO Sensitivity (Recall)** | **78.2%** | Correctly flags **~4 in 5** multi-drug-resistant-organism episodes as elevated-risk *before* culture confirmation — a **+52.7% relative improvement** over blind guessing on the highest-stakes patient subgroup |
| **Temporal (AUC) Drift** | **1.29%** over 18 simulated months | Model-quality decay stays **≥3.8× inside** the 5% retention target — supports a realistic **monthly retrain cadence** instead of costly full rebuilds |
| **Training Throughput** | **~seconds** per 13,500-row fold | A full retrain cycle is cheap enough to run on an **antibiogram-like monthly cadence** without dedicated infrastructure |
| **Explainability Coverage** | **100%** of predictions | Every risk score ships with SHAP / gain-based feature attribution — no "black box" risk score for a pharmacist to sign off on blind |
| **Time-to-Signal** | **Hour 0–1** (vs. Hour 48–72 for culture ID) | Delivers a resistance estimate **~48–72 hours earlier** than the standard-of-care lab result, inside the actual empiric-prescribing decision window |

### Framed as a stewardship business case

- **Decision-support coverage, not diagnosis:** at 0.717 mean AUC-ROC, the model gives every empiric-prescribing decision a *quantified, auditable* second opinion where today there is none — turning a subjective judgment call into a documented, reviewable data point for the stewardship committee.
- **Safety-first thresholding:** decision thresholds were tuned to prioritize catching multi-drug-resistant organisms (78.2% recall) rather than defaulting to 0.5 — reflecting the real clinical asymmetry that a missed MDRO case is far costlier than a false alarm requiring pharmacist review.
- **Low total cost of ownership:** an 11-feature, chart-native input schema and a sub-minute inference pipeline mean no new instrumentation, no waiting on labs, and a retrain cost measured in seconds — a stewardship program could operate this on commodity infrastructure.
- **Governance-ready from day one:** stated intended use, disclosed limitations, and a "pharmacist review required" banner are permanent parts of the interface, not addenda — reducing the compliance and audit lift for any team evaluating this for a real pilot.

---

## 🩺 The Clinical Problem

```
Hour 0                Hour 0–1                    Hour 24–48              Hour 48–72
Patient presents  →   Empiric antibiotic chosen  → Culture growth        → Susceptibility
w/ suspected            (no lab result yet,           identified            panel finalized;
infection                AMR Predictor fires here)                          therapy adjusted
```

- The WHO names antimicrobial resistance among the most serious global health threats, tracked via its GLASS surveillance network across 100+ countries.
- Global Burden of Disease researchers project AMR could be linked to **~10 million deaths/year by 2050** if current trends continue unchecked.
- Priority pathogens (e.g., carbapenem-resistant Gram-negatives) are outpacing the antibiotic development pipeline — every prescribing decision is higher-stakes than the last.

AMR Predictor sits at the **patient level, in real time** — a layer beneath population-level surveillance and facility-level quarterly antibiograms, closing a gap neither of those can close.

---

## 🏗️ System Architecture

```
amr-resistance-predictor/
├── data/
│   ├── raw/                # Unprocessed historical EHR extracts (git-ignored)
│   ├── processed/          # Cleaned, feature-engineered parquet files
│   └── schema.yaml         # Data type validation rules and missingness thresholds
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory analysis of pathogen distributions & missingness
│   └── 02_baseline.ipynb   # Baseline model training and hyperparameter search
├── src/
│   ├── data_loader.py      # Automated extraction and schema validation
│   ├── features/
│   │   ├── temporal.py     # Rolling 30/90/365-day patient-history aggregations
│   │   └── engineering.py  # Categorical encoding, target encoding, missingness flags
│   ├── models/
│   │   ├── train.py        # Multi-label model training orchestration & checkpointing
│   │   └── evaluate.py     # Metrics, confusion matrices, per-class threshold tuning
│   └── utils/               # Config parsing, structured logging, helpers
├── dashboard/
│   └── app.py               # Interactive Streamlit dashboard for clinical review
├── model_card.md            # Intended use, performance limits, audit trail
├── requirements.txt         # Pinned dependencies
└── README.md                 # You are here
```

---

## 🧠 Modeling Approach

| Design decision | Choice | Rationale |
|---|---|---|
| **Problem framing** | Multi-label, binary-relevance | A single isolate can resist multiple drug classes simultaneously — the label space is a 6-bit vector, not a single class pick |
| **Algorithm** | 6× independent **XGBoost** binary classifiers | Native missing-value handling, no scaling/embedding needed, gain-based interpretability, industry-standard for tabular clinical risk scoring |
| **Feature set** | 11 chart-native inputs (exposure history, device presence, note-derived flags, calendar month) | Every input is already in the chart at triage or derivable from note text in seconds — **nothing waits on a lab** |
| **Pathogen identity** | **Deliberately excluded** | Including it would leak the future and inflate AUC — by the time an organism is identified, the empiric decision is already made. Every reported AUC (0.696–0.724) reflects the harder, honest task of predicting from risk factors alone |
| **Decision thresholds** | Tuned per class via Youden Index | Not a default 0.5 cutoff — balances sensitivity/specificity against real clinical error costs, independently per antibiotic class |
| **Validation strategy** | Static 75/25 split **+** 3-fold rolling-origin backtest (months 12→30) | Guards against both a lucky snapshot *and* undetected antibiogram drift — the real test of deployability |

**Shared hyperparameter configuration** (deliberately conservative, shallow-tree):

| Param | Value | Purpose |
|---|---|---|
| `n_estimators` | 250 | Boosting rounds |
| `max_depth` | 4 | Shallow trees — resists overfitting on an 11-feature space |
| `learning_rate` | 0.08 | Conservative shrinkage |
| `subsample` / `colsample_bytree` | 0.85 / 0.85 | Row/column sampling per tree |
| `objective` | `binary:logistic` | Calibrated probability output |
| `eval_metric` | `auc` | Matches the primary reporting metric |
| `random_state` | 42 | Reproducibility |

---

## 📊 Results in Detail

### Per-antibiotic-class discrimination (held-out test set, n = 4,500)

| Antibiotic class | AUC-ROC | Prevalence | vs. 0.50 baseline |
|---|---|---|---|
| Cephalosporins | **0.7242** | 62.2% | +44.8% |
| Glycopeptides | **0.7234** | 35.4% | +44.7% |
| Fluoroquinolones | **0.7215** | 57.1% | +44.3% |
| Aminoglycosides | **0.7206** | 43.8% | +44.1% |
| Carbapenems | **0.7157** | 31.9% | +43.1% |
| Penicillins | **0.6959** | 70.4% | +39.2% |
| **Mean** | **0.717** | — | **+43.4%** |

Penicillins remain the hardest class to discriminate — consistent with the widest, most heterogeneous resistance mechanisms of the six.

### Temporal robustness — rolling-origin backtest

| Fold | Train origin | Test window | n (episodes) | AUC-ROC |
|---|---|---|---|---|
| 1 | Month 12 | Months 12–18 | 2,944 | 0.7153 |
| 2 | Month 18 | Months 18–24 | 3,003 | 0.7064 |
| 3 | Month 24 | Months 24–30 | 2,995 | 0.7061 |

**First-to-last drift: 1.29%** — well inside the ≤5% retention target, and evidence the model's feature–risk relationships generalize beyond its training window rather than memorizing it.

---

## 🗂️ Dataset & Cohort

| Attribute | Value |
|---|---|
| Total culture episodes | 18,000 |
| Simulated calendar span | 36 months |
| Distinct pathogen species | 8 (6 of 8 on the WHO 2024 Bacterial Priority Pathogens List) |
| Antibiotic resistance classes | 6 |
| Composite MDRO-positive rate | 60.2% |
| Train / held-out test split | 13,500 / 4,500 (75/25) |

**Feature vector — 11 inputs, 4 categories:**

- **Exposure history (3):** `prior_abx_90d`, `recent_hosp`, `nursing_home`
- **Device presence (4):** `device_catheter`, `device_central_line`, `device_ventilator`, `icu_admission`
- **Note-derived flags (3):** `note_flag_prior_mdro`, `note_flag_recent_travel`, `note_flag_immunosuppressed`
- **Temporal (1):** `month`

Top predictive driver (carbapenem classifier, generalizes across all six): `note_flag_prior_mdro` — a documented prior MDRO history carries **~2.3×** the gain-based weight of the next strongest feature (`recent_hosp`), consistent with clinical intuition that resistance recurs in the same patient.

---

## 🔍 Data Transparency & Disclosure

This cohort is **synthetic by design** — generated to reproduce the statistical shape of a real hospital microbiology + EHR feed (monthly volume, class-conditional risk-factor correlation, deliberate antibiogram drift) **without touching any real patient data**, removing privacy/IRB constraints entirely while keeping temporal validation meaningful.

> **This is stated as a headline limitation, not a footnote.** Every metric in this README is scoped to this synthetic cohort and is **not** a claim of real-world clinical performance. Treating this disclosure as first-class is itself part of demonstrating responsible ML practice for a clinical-adjacent tool.

---

## ⚙️ Installation & Local Development

**Prerequisites:** Python 3.10+, Git, a virtual environment manager.

```bash
# 1. Clone the repository
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/ANTIMICROBIAL_RESISTANCE_PREDICTOR.git
cd ANTIMICROBIAL_RESISTANCE_PREDICTOR

# 2. Create and activate an isolated virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

```bash
# Feature engineering — raw EHR extract → temporal feature store
python src/features/engineering.py --config data/schema.yaml

# Evaluation pipeline — including rolling-origin temporal backtest
python src/models/evaluate.py --mode temporal

# Interactive clinical review dashboard
streamlit run dashboard/app.py
```

---

## ⚖️ Responsible AI & Operational Guardrails

Clinical ML software requires strict validation, transparency, and governance before any live deployment. This project treats the following as **first-class design choices, not afterthoughts**:

- **Human-in-the-loop by design** — every prediction is a decision-support input reviewed by a stewardship pharmacist; the dashboard states this permanently, not in fine print.
- **Explainability by default** — every risk score ships with SHAP values quantifying exactly which chart features (e.g., recent fluoroquinolone exposure, ICU length of stay) drove it.
- **Honest baseline reporting** — every headline number is shown alongside the 0.50 blind-guess floor, never presented in isolation.
- **Fairness auditing** — outputs are checked across demographic groups, age bands, and hospital units for systematic disparities in predictive error.
- **No stability claim without evidence** — the 1.29% drift figure exists *because* of the rolling-origin backtest (Section 06 of the accompanying portfolio deck), not asserted on faith.
- **Intended use, stated explicitly:** clinical decision support only. **Does not** replace professional medical judgment, microbiological culture confirmation, or infectious disease specialist consultation. Final clinical decisions remain the sole responsibility of licensed practitioners.

### Known limitations

- **Synthetic cohort** — not real-world clinical data (see [Data Transparency](#-data-transparency--disclosure)).
- **Note-derived flags** use rule-based regex extraction — fast and auditable, but brittle to phrasing variance; not a fine-tuned clinical language model.
- **Performance is scoped** to the evaluated cohort and should not be extrapolated to other populations without re-validation.

### Roadmap

- Real **ClinicalBERT**-based note extraction to replace regex flagging
- Live antibiogram feed for continuous, rather than simulated, retraining
- **CDS (clinical decision support)** hooks at order-entry time
- Exploration of whole-genome-sequencing (WGS) resistance markers

---

## 🧩 Skills Demonstrated

`Multi-label clinical ML` · `Temporal / rolling-origin validation` · `Feature engineering under data-leakage constraints` · `Gradient-boosted tree modeling (XGBoost / LightGBM)` · `Model documentation & responsible-AI framing` · `Stakeholder-ready dashboarding (Streamlit)` · `End-to-end pipeline design`

---

## 📄 License

Distributed under the **MIT License**. See the `LICENSE` file for full terms.

## 👤 Author

**Anthony Chinedu Echem**
Machine Learning Engineer & Data Scientist
📫 Portfolio contact available on request

---

<p align="center"><em>A single structured-data model, retrained monthly on routine EHR fields, keeps clinically useful discrimination stable for at least 30 months without full retraining — a realistic stewardship-program deployment story.</em></p>
