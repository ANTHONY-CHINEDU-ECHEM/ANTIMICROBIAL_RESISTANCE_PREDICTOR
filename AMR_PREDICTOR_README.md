# AMR Predictor: An Antimicrobial Resistance Risk and Stewardship System

**A production grade, multi label machine learning system that predicts patient specific antibiotic resistance probabilities at the point of care, before culture results exist.**

[![Status](https://img.shields.io/badge/status-reference%20implementation-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Model](https://img.shields.io/badge/model-XGBoost%20%7C%20LightGBM-orange)]()

> **Built by [Anthony Chinedu Echem](#license-and-author)**, Machine Learning Engineer and Data Scientist
> An end to end machine learning portfolio project covering data design, modeling, temporal validation, and deployment.

<img width="1307" height="816" alt="AMR Predictor: Interactive Stewardship Dashboard" src="https://github.com/user-attachments/assets/794ac2e8-5db7-4317-8e57-5122603ca020" />

<p align="center"><em>The reference dashboard: headline key performance indicators, per class discrimination, rolling origin backtest stability, cohort composition, and a permanent model card summary, all in one pharmacist facing view.</em></p>

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quantified Impact: What the Numbers Mean in Practice](#quantified-impact-what-the-numbers-mean-in-practice)
3. [The Clinical Problem](#the-clinical-problem)
4. [System Architecture](#system-architecture)
5. [Modeling Approach](#modeling-approach)
6. [Results in Detail](#results-in-detail)
7. [Dataset and Cohort](#dataset-and-cohort)
8. [Data Transparency and Disclosure](#data-transparency-and-disclosure)
9. [Installation and Local Development](#installation-and-local-development)
10. [Responsible AI and Operational Guardrails](#responsible-ai-and-operational-guardrails)
11. [Known Limitations](#known-limitations)
12. [Roadmap](#roadmap)
13. [Skills Demonstrated](#skills-demonstrated)
14. [License and Author](#license-and-author)

---

## Executive Summary

Empiric antibiotic therapy has to be chosen before laboratory culture and susceptibility results exist, a 48 to 72 hour blind spot in which clinicians rely on judgment, quarterly hospital antibiograms, and patient history alone. Choosing too broad a spectrum accelerates resistance across the hospital network; choosing too narrow a spectrum risks treatment failure. AMR Predictor closes that gap with a per patient, per antibiotic class resistance probability generated at the exact moment a culture is ordered, using only data already sitting in the chart.

This repository is a reference implementation: a synthetic but realistic 18,000 episode cohort, a fully reproducible modeling pipeline, rigorous temporal, rather than purely random, validation, and a pharmacist facing dashboard, packaged the way a real stewardship program deliverable would be, including its limitations stated up front rather than buried in an appendix.

---

## Quantified Impact: What the Numbers Mean in Practice

> **Methodology note.** All figures below are computed on a held out or rolling origin test split of the project's synthetic 18,000 episode cohort (see [Data Transparency and Disclosure](#data-transparency-and-disclosure)). They are reported here as modeled, illustrative operational impact, the kind of business case a stewardship program would build before a pilot, and not as claims of realized clinical outcomes. Every number is benchmarked against an honest, no information baseline, never an inflated straw man comparison.

| Metric | Result | What it means operationally |
|---|---|---|
| Mean AUC ROC | 0.717 across 6 resistance classes | A discrimination lift of 43.4 percent over the 0.50 blind guess floor. The model reliably rank orders which patients are more likely to be resistant, turning a coin flip into a triageable risk signal. |
| MDRO sensitivity, that is, recall for multi drug resistant organisms | 78.2 percent | Correctly flags roughly 4 in 5 multi drug resistant organism episodes as elevated risk before culture confirmation, a relative improvement of 52.7 percent over blind guessing on the highest stakes patient subgroup. |
| Temporal AUC drift | 1.29 percent over 18 simulated months | Model quality decay stays more than 3.8 times inside the 5 percent retention target, supporting a realistic monthly retrain cadence instead of costly full rebuilds. |
| Training throughput | Approximately seconds per 13,500 row fold | A full retrain cycle is cheap enough to run on an antibiogram like monthly cadence without dedicated infrastructure. |
| Explainability coverage | 100 percent of predictions | Every risk score ships with SHAP based, gain derived feature attribution, so there is no black box risk score for a pharmacist to sign off on blind. |
| Time to signal | Hour 0 to hour 1, compared with hour 48 to hour 72 for a culture identification | Delivers a resistance estimate roughly 48 to 72 hours earlier than the standard of care lab result, inside the actual empiric prescribing decision window. |

### Framed as a stewardship business case

- **Decision support coverage, not diagnosis.** At a mean AUC ROC of 0.717, the model gives every empiric prescribing decision a quantified, auditable second opinion where today there is none, turning a subjective judgment call into a documented, reviewable data point for the stewardship committee.
- **Safety first thresholding.** Decision thresholds were tuned to prioritize catching multi drug resistant organisms, achieving 78.2 percent recall, rather than defaulting to a 0.5 cutoff. This reflects the real clinical asymmetry that a missed multi drug resistant organism case is far costlier than a false alarm that simply requires pharmacist review.
- **A low total cost of ownership.** An 11 feature, chart native input schema and a sub minute inference pipeline mean no new instrumentation, no waiting on labs, and a retrain cost measured in seconds. A stewardship program could operate this on commodity infrastructure.
- **Governance ready from day one.** A stated intended use, disclosed limitations, and a pharmacist review required banner are permanent parts of the interface, not an addendum, reducing the compliance and audit burden for any team evaluating this for a real pilot.

---

## The Clinical Problem

```
Hour 0                    Hour 0 to 1                       Hour 24 to 48              Hour 48 to 72
Patient presents    ->    Empiric antibiotic chosen    ->   Culture growth        ->   Susceptibility
with suspected             (no lab result yet;                identified                 panel finalized;
infection                   AMR Predictor fires here)                                     therapy adjusted
```

- The World Health Organization names antimicrobial resistance among the most serious global health threats, tracked through its GLASS surveillance network across more than 100 countries.
- Global Burden of Disease researchers project that antimicrobial resistance could be linked to approximately 10 million deaths per year by 2050 if current trends continue unchecked.
- Priority pathogens, such as carbapenem resistant Gram negative organisms, are outpacing the antibiotic development pipeline, meaning every prescribing decision carries higher stakes than the last.

AMR Predictor operates at the patient level, in real time, a layer beneath population level surveillance and facility level quarterly antibiograms, closing a gap that neither of those existing tools can close on its own.

---

## System Architecture

```
amr_resistance_predictor/
├── data/
│   ├── raw/                 # Unprocessed historical EHR extracts (excluded from version control)
│   ├── processed/           # Cleaned, feature engineered parquet files
│   └── schema.yaml          # Data type validation rules and missingness thresholds
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory analysis of pathogen distributions and missingness
│   └── 02_baseline.ipynb    # Baseline model training and hyperparameter search
├── src/
│   ├── data_loader.py       # Automated extraction and schema validation
│   ├── features/
│   │   ├── temporal.py      # Rolling 30, 90, and 365 day patient history aggregations
│   │   └── engineering.py   # Categorical encoding, target encoding, missingness flags
│   ├── models/
│   │   ├── train.py         # Multi label model training orchestration and checkpointing
│   │   └── evaluate.py      # Metrics, confusion matrices, per class threshold tuning
│   └── utils/                # Config parsing, structured logging, and shared helpers
├── dashboard/
│   └── app.py                 # Interactive Streamlit dashboard for clinical review
├── model_card.md               # Intended use, performance limits, and audit trail
├── requirements.txt             # Pinned dependencies
└── README.md                     # This document
```

---

## Modeling Approach

| Design decision | Choice | Rationale |
|---|---|---|
| Problem framing | Multi label, binary relevance | A single isolate can resist multiple drug classes simultaneously, so the label space is a 6 bit vector rather than a single class pick. |
| Algorithm | Six independent XGBoost binary classifiers | Native missing value handling, no scaling or embedding required, gain based interpretability, and an industry standard choice for tabular clinical risk scoring. |
| Feature set | 11 chart native inputs, covering exposure history, device presence, note derived flags, and calendar month | Every input is already in the chart at triage or derivable from note text within seconds, so nothing waits on a lab result. |
| Pathogen identity | Deliberately excluded | Including it would leak information from the future and artificially inflate the AUC score, since by the time an organism is identified, the empiric decision has already been made. Every reported AUC, ranging from 0.696 to 0.724, reflects the harder, honest task of predicting from risk factors alone. |
| Decision thresholds | Tuned per class using the Youden Index | Not a default 0.5 cutoff; this approach balances sensitivity and specificity against real clinical error costs, independently for each antibiotic class. |
| Validation strategy | A static 75 to 25 split combined with a 3 fold rolling origin backtest spanning months 12 through 30 | Guards against both a lucky snapshot and undetected antibiogram drift, representing the real test of deployability. |

**Shared hyperparameter configuration**, deliberately conservative and using shallow trees:

| Parameter | Value | Purpose |
|---|---|---|
| `n_estimators` | 250 | The number of boosting rounds |
| `max_depth` | 4 | Shallow trees that resist overfitting on an 11 feature space |
| `learning_rate` | 0.08 | Conservative shrinkage applied at each boosting round |
| `subsample` and `colsample_bytree` | 0.85 and 0.85 | Row and column sampling applied per tree |
| `objective` | `binary:logistic` | Produces a calibrated probability output |
| `eval_metric` | `auc` | Matches the primary reporting metric used throughout this document |
| `random_state` | 42 | Ensures reproducibility |

---

## Results in Detail

### Per antibiotic class discrimination, held out test set, n equals 4,500

| Antibiotic class | AUC ROC | Prevalence | Improvement over the 0.50 baseline |
|---|---|---|---|
| Cephalosporins | 0.7242 | 62.2 percent | 44.8 percent |
| Glycopeptides | 0.7234 | 35.4 percent | 44.7 percent |
| Fluoroquinolones | 0.7215 | 57.1 percent | 44.3 percent |
| Aminoglycosides | 0.7206 | 43.8 percent | 44.1 percent |
| Carbapenems | 0.7157 | 31.9 percent | 43.1 percent |
| Penicillins | 0.6959 | 70.4 percent | 39.2 percent |
| Mean across all six classes | 0.717 | not applicable | 43.4 percent |

Penicillins remain the hardest class to discriminate, which is consistent with having the widest and most heterogeneous resistance mechanisms of the six classes evaluated.

### Temporal robustness: rolling origin backtest

| Fold | Train origin | Test window | Episodes (n) | AUC ROC |
|---|---|---|---|---|
| 1 | Month 12 | Months 12 to 18 | 2,944 | 0.7153 |
| 2 | Month 18 | Months 18 to 24 | 3,003 | 0.7064 |
| 3 | Month 24 | Months 24 to 30 | 2,995 | 0.7061 |

**First to last drift: 1.29 percent**, well inside the target retention threshold of 5 percent or less, and evidence that the model's feature to risk relationships generalize beyond its training window rather than simply memorizing it.

---

## Dataset and Cohort

| Attribute | Value |
|---|---|
| Total culture episodes | 18,000 |
| Simulated calendar span | 36 months |
| Distinct pathogen species | 8, of which 6 appear on the World Health Organization 2024 Bacterial Priority Pathogens List |
| Antibiotic resistance classes | 6 |
| Composite multi drug resistant organism positive rate | 60.2 percent |
| Train and held out test split | 13,500 and 4,500 episodes, a 75 to 25 split |

**Feature vector: 11 inputs across 4 categories**

- Exposure history (3 features): `prior_abx_90d`, `recent_hosp`, `nursing_home`
- Device presence (4 features): `device_catheter`, `device_central_line`, `device_ventilator`, `icu_admission`
- Note derived flags (3 features): `note_flag_prior_mdro`, `note_flag_recent_travel`, `note_flag_immunosuppressed`
- Temporal (1 feature): `month`

The top predictive driver for the carbapenem classifier, a pattern that generalizes across all six classes, is `note_flag_prior_mdro`. A documented prior multi drug resistant organism history carries approximately 2.3 times the gain based weight of the next strongest feature, `recent_hosp`, which is consistent with clinical intuition that resistance tends to recur in the same patient.

---

## Data Transparency and Disclosure

This cohort is synthetic by design, generated to reproduce the statistical shape of a real hospital microbiology and electronic health record feed, including monthly volume, class conditional risk factor correlation, and deliberate antibiogram drift, without touching any real patient data. This removes privacy and institutional review board constraints entirely while keeping temporal validation meaningful.

> **This is stated as a headline limitation, not a footnote.** Every metric in this document is scoped to this synthetic cohort and is not a claim of real world clinical performance. Treating this disclosure as a first class part of the documentation is itself part of demonstrating responsible machine learning practice for a clinically adjacent tool.

---

## Installation and Local Development

**Prerequisites:** Python 3.10 or later, Git, and a virtual environment manager.

```bash
# 1. Clone the repository
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/ANTIMICROBIAL_RESISTANCE_PREDICTOR.git
cd ANTIMICROBIAL_RESISTANCE_PREDICTOR

# 2. Create and activate an isolated virtual environment
python -m venv venv
source venv/bin/activate        # on Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

```bash
# Feature engineering: converts a raw EHR extract into the temporal feature store
python src/features/engineering.py --config data/schema.yaml

# Evaluation pipeline, including the rolling origin temporal backtest
python src/models/evaluate.py --mode temporal

# Interactive clinical review dashboard
streamlit run dashboard/app.py
```

---

## Responsible AI and Operational Guardrails

Clinical machine learning software requires strict validation, transparency, and governance before any live deployment. This project treats the following as first class design choices, not afterthoughts:

- **Human oversight by design.** Every prediction is a decision support input reviewed by a stewardship pharmacist; the dashboard states this permanently, not in fine print.
- **Explainability by default.** Every risk score ships with SHAP values quantifying exactly which chart features, such as recent fluoroquinolone exposure or intensive care unit length of stay, drove it.
- **Honest baseline reporting.** Every headline number is shown alongside the 0.50 blind guess floor, never presented in isolation.
- **Fairness auditing.** Outputs are checked across demographic groups, age bands, and hospital units for systematic disparities in predictive error.
- **No stability claim without evidence.** The 1.29 percent drift figure exists because of the rolling origin backtest documented in this repository, not asserted on faith.
- **Intended use, stated explicitly.** This system provides clinical decision support only. It does not replace professional medical judgment, microbiological culture confirmation, or infectious disease specialist consultation. Final clinical decisions remain the sole responsibility of licensed practitioners.

### Known Limitations

- **A synthetic cohort.** This is not real world clinical data; see [Data Transparency and Disclosure](#data-transparency-and-disclosure) above.
- **Note derived flags use rule based regular expression extraction.** This approach is fast and auditable, but brittle to phrasing variance, and is not a fine tuned clinical language model.
- **Performance is scoped to the evaluated cohort** and should not be extrapolated to other populations without re validation.

### Roadmap

- Real ClinicalBERT based note extraction to replace regular expression flagging.
- A live antibiogram feed to support continuous, rather than simulated, retraining.
- Clinical decision support hooks integrated at order entry time.
- Exploration of whole genome sequencing resistance markers as additional model inputs.

---

## Skills Demonstrated

Multi label clinical machine learning, temporal and rolling origin validation, feature engineering under data leakage constraints, gradient boosted tree modeling using XGBoost and LightGBM, model documentation and responsible AI framing, stakeholder ready dashboarding using Streamlit, and end to end pipeline design.

---

## License and Author

This project is distributed under the MIT License. See the `LICENSE` file for full terms.

**Anthony Chinedu Echem**
Machine Learning Engineer and Data Scientist
Portfolio contact available on request

---

<p align="center"><em>A single structured data model, retrained monthly on routine electronic health record fields, keeps clinically useful discrimination stable for at least 30 months without a full retraining cycle: a realistic stewardship program deployment story.</em></p>
