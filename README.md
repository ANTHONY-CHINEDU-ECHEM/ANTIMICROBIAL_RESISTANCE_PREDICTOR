**AMR PREDICTOR: ANTIMICROBIAL RESISTANCE RISK AND STEWARDSHIP SYSTEM**


A production-grade, end-to-end machine learning system designed to predict patient-specific antibiotic resistance probabilities at the point of care. This framework forecasts specific microbial resistance profiles to assist clinical staff in selecting appropriate empiric antibiotic therapy, managing multi-drug resistant organisms, and enforcing institutional antimicrobial stewardship policies.


<img width="1307" height="816" alt="AMR DASHBOARD" src="https://github.com/user-attachments/assets/794ac2e8-5db7-4317-8e57-5122603ca020" />



**PROJECT OVERVIEW AND CLINICAL MOTIVATION**


Empiric antibiotic therapy is the administration of antibacterial medications prior to the availability of definitive laboratory culture and susceptibility results. Initiating therapy quickly is necessary for severe infections, but choosing overly broad-spectrum agents accelerates bacterial resistance across hospital networks. Conversely, choosing insufficient coverage leads to treatment failure, prolonged hospital stays, and increased patient mortality.

The AMR Predictor addresses this clinical trade-off by implementing a multi-label classification system. The system ingests electronic health record data available at the exact timestamp of culture collection to output probability scores for resistance against individual antibiotic classes. These scores provide quantitative decision support to clinicians before laboratory identification protocols conclude.



**KEY PERFORMANCE INDICATORS AND EVALUATION METRICS**



The modelling pipeline uses rigorous performance metrics evaluated across distinct testing paradigms to quantify clinical safety and technical reliability:Predictive Discrimination: The system achieved a mean Area Under the Receiver Operating Characteristic curve of 0.717 across all evaluated pathogen-drug combinations.  Clinical Safety and Sensitivity: The decision threshold tuning prioritised multi-drug resistant organism sensitivity, achieving 78.2% sensitivity to minimize false negative classifications for high-risk resistances.  Temporal Stability: Sequential validation windows demonstrated a performance drift of only 1.29% over time, indicating that the feature representation resists degradation caused by changing patient populations.  Calibration Performance: Predicted probabilities undergo post-processing calibration to ensure that output risk scores match empirical frequencies, allowing clinicians to interpret output percentages as actual statistical likelihoods.



**SYSTEM ARCHITECTURE AND DIRECTORY STRUCTURE**


The repository adheres to software engineering standards for modularity, separating data ingestion pipelines, feature extraction scripts, model training routines, and deployment artifacts:

amr-resistance-predictor/


├── data/


│   ├── raw/                # Unprocessed historical electronic health record extracts (git-ignored)

│   ├── processed/          # Cleaned, feature-engineered parquet files

│   └── schema.yaml         # Data type validation rules and missingness thresholds

├── notebooks/

│   ├── 01_eda.ipynb        # Exploratory data analysis of pathogen distributions and missingness

│   └── 02_baseline.ipynb   # Baseline model training and hyperparameter search scripts

├── src/

│   ├── __init__.py

│   ├── data_loader.py      # Automated database extraction and schema validation routines

│   ├── features/

│   │   ├── temporal.py     # Rolling window calculations for patient history and lab trends

│   │   └── engineering.py  # Categorical encoding, target encoding, and missingness indicator creation

│   ├── models/

│   │   ├── train.py        # Multi-label model training orchestration and checkpoint saving

│   │   └── evaluate.py     # Metric computation, confusion matrix generation, and threshold tuning

│   └── utils/              # Configuration parsers, structured logging setup, and helper utilities

├── dashboard/

│   └── app.py              # Interactive Streamlit application for clinical review and local inference

├── model_card.md           # Formal documentation outlining intended use, performance limits, and audits

├── requirements.txt        # Explicitly pinned python package dependencies

└── README.md               # Comprehensive system documentation



**DATA ENGINEERING AND TEMPORAL VALIDATION STRATEGY**


To eliminate data leakage, which occurs when future information accidentally influences historical training samples, the project replaces random cross-validation with a strictly chronological temporal split strategy.

Temporal Split Architecture: The training dataset comprises historical patient records from early years, the validation set comprises intermediate data for hyperparameter tuning, and the testing set comprises prospective data from the final chronological year to simulate live deployment.

Rolling Patient History Aggregation: The feature engineering pipeline computes rolling aggregations of patient health records over sliding windows of 30 days, 90 days, and 365 days. These windows capture prior hospital admissions, previous antibiotic therapy exposures, invasive device placements, and ward-level resistance prevalence.

Missing Data Management: Clinical datasets frequently contain missing laboratory values or unmeasured vitals due to irregular ordering practices. The pipeline handles missing data by generating explicit binary missingness indicator variables and coupling them with structured imputation strategies.




**MACHINE LEARNING PIPELINE AND MODELLING METHODOLOGY**


**Multi-Label Problem Formulation:** Because a single bacterial isolate can demonstrate concurrent resistance or susceptibility to multiple distinct antibiotics, the modelling framework implements independent binary classifiers per antibiotic target or uses gradient-boosted multi-output configurations.

**Algorithm Selection:** The pipeline utilizes gradient-boosted decision tree frameworks via LightGBM and XGBoost. These algorithms handle missing values natively, model non-linear interactions between clinical features, and scale efficiently across large tabular datasets.

**Decision Threshold Optimization:** Classification thresholds are not set to a default value of 0.5. Instead, thresholds are optimized independently for each antibiotic target on the validation set to maximize the Youden Index, balancing sensitivity and specificity based on clinical error costs.


**INSTALLATION AND LOCAL DEVELOPMENT**

**PREREQUISITES**

- Python version 3.10 or higher

- Git version control system

- A local virtual environment manager

**STEP BY STEP INSTALLATION INSTRUCTIONS**

- CLONE THE REPOSITORY FROM THE REMOTE SERVER:

git clone https://github.com/ANTHONY-CHINEDU-ECHEM/ANTIMICROBIAL_RESISTANCE_PREDICTOR.git
cd ANTIMICROBIAL_RESISTANCE_PREDICTOR

- CREATE AND ACTIVATE AN ISOLATED VIRTUAL ENVIRONMENT

  python -m venv venv
source venv/bin/activate  # On Windows command prompt: venv\Scripts\activate

- UPGRADE pip AND INSTALL ALL REQUIRED PROJECT DEPENDENCIES

  pip install --upgrade pip
pip install -r requirements.txt


**EXECUTION AND USAGE GUIDE**

**1. DATA PROCESSING AND FEATURE ENGINEERING EXECUTION**

To process raw data templates according to the schema rules and generate temporal feature stores, run the following command:


python src/features/engineering.py --config data/schema.yaml

**2. EVALUATION PIPELINE EXECUTION**

python src/models/evaluate.py --mode temporal

**3. INTERACTIVE CLINICAL DASHBOARD EXECUTION**

streamlit run dashboard/app.py

**RESPONSIBLE AI AND OPERATIONAL LIMITATIONS**


-Clinical machine learning software requires strict adherence to validation, transparency, and governance protocols before deployment in live hospital environments:


- **Interpretability and Explanation:** Every individual risk prediction is paired with SHAP values. These values quantify the exact contribution of specific electronic health record features, such as recent fluoroquinolone exposure or intensive care unit length of stay, toward the final output probability.


- **Fairness and Subgroup Auditing:** The model outputs are audited across demographic categories, age groups, and hospital units to check for systematic disparities in predictive error rates and prevent bias amplification.

- **Intended Use Limitations:** This software functions exclusively as a clinical decision support tool. It does not replace professional medical judgment, microbiological culturing verification, or infectious disease specialist consultation. Final clinical management decisions remain the sole responsibility of licensed healthcare practitioners.


**License**

This software project is distributed under the terms of the MIT License. Review the repository LICENSE file for full legal terms and conditions.



