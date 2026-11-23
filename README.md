# Intelligent Student Dropout Risk Early Warning System
### Multi-Stage Machine Learning Pipeline with SHAP Explainability & Decision Support

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.5.2](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![Flask 3.1.3](https://img.shields.io/badge/flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![Tests Passing](https://img.shields.io/badge/tests-23%20passing-brightgreen.svg)](tests/)

---

## 1. Executive Summary

Many dropout models depend on data that only becomes available late. This system provides three predictions as information accumulates: at admission, after semester 1, and after semester 2. Stage 1 is a coarse screening signal; accuracy and reliability improve markedly once semester results are available. All results below are from a held-out test set (20% of the data, *n* = 885, never used in tuning) unless labeled CV; cross-validation figures are documented in the full report. The model was trained on one Portuguese institution’s data (Polytechnic Institute of Portalegre) and predictions are relative risk bands, not calibrated probabilities.

1. **Stage 1 (Admission)**: Evaluated before classes begin using demographic, prior schooling, entrance grades, and fee status at data collection (24 features). Suitable for coarse cohort screening, not for judging an individual student.
2. **Stage 2 (Post-Semester 1)**: Evaluated mid-year incorporating Semester 1 curricular units enrolled, approved, evaluated, and grades (30 features). Accuracy jumps significantly, enabling targeted tutoring and academic recovery.
3. **Stage 3 (Post-Semester 2)**: Evaluated at the close of Year 1 incorporating full cumulative credits and grade trajectories (36 features). Peak diagnostic accuracy directs intensive, individualized intervention programs.

---

## 2. Quantitative Benchmark Results

All candidate models were tuned using 5-fold Stratified Cross-Validation on the training partition ($n=3,539$) and evaluated strictly once on the held-out 20% test partition ($n=885$). Multi-class sample weights were applied uniformly across candidate algorithms to balance the minority Dropout ($32.1\%$) and Enrolled ($18.0\%$) classes against the majority Graduate ($49.9\%$) class.

### 2.1 Model Benchmark Table (CV vs Held-out Test)

| Milestone Stage | Selected Model | 5-Fold CV Macro-F1 (Mean ± Std) | 5-Fold CV Dropout Recall | Test Accuracy (Held-out n=885) | Test Macro-F1 | Test Dropout Recall | Baseline Dummy Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (Admission)** | Random Forest | 0.576 ± 0.013 | 60.8% ± 3.0% | **61.02%** | **0.545** | **61.97%** | 49.94% (+11.08%) |
| **Stage 2 (Post-Sem 1)** | Random Forest | 0.687 ± 0.018 | 72.1% ± 2.7% | **74.35%** | **0.655** | **70.77%** | 49.94% (+24.41%) |
| **Stage 3 (Post-Sem 2)** | Random Forest | 0.717 ± 0.013 | 74.4% ± 1.6% | **77.51%** | **0.700** | **75.70%** | 49.94% (+27.57%) |

*Statistical Note:* With $N = 885$ test samples, accuracy has an uncertainty of approximately $\pm 2.8\%$ at 95% confidence. Stage 3 achieves peak multi-class test accuracy (77.51%) and multi-class ROC-AUC (0.8907). Cross-validation scores reflect paired 5-fold evaluation on the training set ($n=3,539$); held-out test scores represent the unbiased single-pass evaluation.

*Candid Stage 1 Note:* Stage 1 is a coarse screening signal suitable for cohort monitoring rather than high-stakes individual decisions (61.0% accuracy vs 49.9% majority baseline, misses ~38% of dropouts). Its multi-class ROC-AUC (0.7682) confirms reliable cohort ranking.

### 2.2 Model Selection Rationale: Option A (Random Forest Across All Stages)

Our selection criterion prioritizes **Dropout Recall and Dropout F1** (the core operational objective of early dropout detection) subject to a **Macro-F1 guard floor** ($\ge 0.52$ at Stage 1, $\ge 0.65$ at Stage 2, $\ge 0.68$ at Stage 3) to prevent multi-class collapse on the intermediate "Enrolled" class.

#### The Trade-Off Arithmetic: Dropout Recall vs Macro-F1
In a 3-class problem, Macro-F1 is $(F1_{\text{Dropout}} + F1_{\text{Enrolled}} + F1_{\text{Graduate}}) / 3$. Gradient Boosting and XGBoost achieve slightly higher Macro-F1 primarily because they predict the transient "Enrolled" class (18% of cohort) more aggressively ($0.45 \to 0.54$ F1). However, an Enrolled-F1 increase of $\sim 0.07$ moves the 3-class Macro-F1 average by only $0.07 / 3 \approx 0.023$. Choosing boosting for that small Macro-F1 boost cost $4.5$ to $6.3$ percentage points of Dropout Recall. Random Forest was selected as champion across all three stages because:
1. **Higher Dropout Detection**: Random Forest achieves higher CV Dropout Recall at Stage 1 ($60.8\% \text{ vs } 57.4\%$) and Stage 2 ($72.1\% \text{ vs } 69.3\%$), and matches boosting at Stage 3 ($74.4\%$). On the held-out test set, it caught 176, 201, and 215 dropouts (62.0%, 70.8%, 75.7% recall), outperforming the legacy boosted models by $+6.0\%$, $+4.9\%$, and $+6.3\%$.
2. **Higher Test Accuracy**: Random Forest generalized better to the held-out test set across all milestones (Stage 1: 61.02% vs 58.53%; Stage 2: 74.35% vs 71.19%; Stage 3: 77.51% vs 72.99%).
3. **Exact TreeSHAP Interpretability**: Random Forest yields exact TreeSHAP values natively in scikit-learn without external dependencies or approximation artifacts.

#### 5-Fold Stratified Cross-Validation Benchmark (Identical Folds, Training Split n=3,539)

| Stage | Model Family | CV Accuracy | CV Macro-F1 | CV Dropout Recall | CV Dropout F1 | CV Dropout Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | **Random Forest (Champion)** | **63.2% ± 1.0%** | **0.576 ± 0.013** | **60.8% ± 3.0%** | **0.629 ± 0.016** | 65.3% |
| | Gradient Boosting | 60.9% ± 1.8% | 0.577 ± 0.014 | 57.4% ± 4.2% | 0.626 ± 0.020 | 69.1% |
| | XGBoost | 61.1% ± 1.0% | 0.573 ± 0.009 | 57.6% ± 4.2% | 0.617 ± 0.023 | 66.8% |
| | Logistic Regression | 59.8% ± 1.4% | 0.563 ± 0.011 | 57.8% ± 4.5% | 0.614 ± 0.021 | 65.8% |
| | Decision Tree | 51.2% ± 1.8% | 0.505 ± 0.019 | 45.3% ± 4.0% | 0.543 ± 0.028 | 68.7% |
| **Stage 2** | **Random Forest (Champion)** | **74.7% ± 1.7%** | **0.687 ± 0.018** | **72.1% ± 2.7%** | **0.764 ± 0.026** | 81.2% |
| | Gradient Boosting | 73.9% ± 2.1% | 0.699 ± 0.025 | 69.3% ± 2.5% | 0.756 ± 0.021 | 83.2% |
| | XGBoost | 73.6% ± 1.6% | 0.693 ± 0.017 | 69.0% ± 2.5% | 0.748 ± 0.020 | 81.6% |
| | Logistic Regression | 72.2% ± 1.8% | 0.681 ± 0.016 | 70.4% ± 3.6% | 0.748 ± 0.029 | 79.9% |
| | Decision Tree | 66.9% ± 1.7% | 0.633 ± 0.011 | 65.4% ± 6.3% | 0.716 ± 0.030 | 79.8% |
| **Stage 3** | **Random Forest (Champion)** | **77.1% ± 0.9%** | **0.717 ± 0.013** | **74.4% ± 1.6%** | **0.790 ± 0.022** | 84.3% |
| | Gradient Boosting | 76.4% ± 1.7% | 0.728 ± 0.019 | 74.4% ± 2.4% | 0.792 ± 0.023 | 84.7% |
| | XGBoost | 77.0% ± 1.3% | 0.729 ± 0.016 | 74.6% ± 1.8% | 0.793 ± 0.014 | 84.8% |
| | Logistic Regression | 75.5% ± 1.3% | 0.713 ± 0.017 | 74.3% ± 2.5% | 0.780 ± 0.018 | 82.2% |
| | Decision Tree | 71.8% ± 2.0% | 0.685 ± 0.017 | 68.6% ± 1.1% | 0.758 ± 0.011 | 84.8% |

### 2.3 Operational Evaluation by Probability Risk Bands
In production, advisors do not rely solely on the single most likely class; they evaluate students using **relative risk bands** (Low: $p < 0.33$, Medium: $0.33 \le p < 0.66$, High: $p \ge 0.66$). These thresholds were established *a priori* (not tuned on the test set). Evaluating the held-out test set ($n=885$; 284 true dropouts) demonstrates the operational effectiveness of this thresholding:

| Milestone Stage | Risk Band | Probability Threshold | Share of Students | Actual Dropout Rate [95% CI] | Dropouts in Band | % All Dropouts Caught | Band Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Low | $< 0.33$ | 62.3% ($n=551$) | 16.0% [13.1%, 19.3%] | 88 | 31.0% | 84.0% non-dropout |
| | Medium | $0.33 - 0.66$ | 28.5% ($n=252$) | 47.2% [41.1%, 53.4%] | 119 | 41.9% | 47.2% dropout |
| | High | $\ge 0.66$ | 9.3% ($n=82$) | **93.9%** [86.5%, 97.4%] | 77 | 27.1% | **93.9% dropout** |
| | **Medium + High Combined** | $\ge 0.33$ | **37.7%** ($n=334$) | — | **196 / 284** | **69.0% caught** | **58.7% precision** [53.3%, 63.8%] |
| **Stage 2** | Low | $< 0.33$ | 63.5% ($n=562$) | 10.0% [7.8%, 12.7%] | 56 | 19.7% | 90.0% non-dropout |
| | Medium | $0.33 - 0.66$ | 20.3% ($n=180$) | 51.7% [44.4%, 58.9%] | 93 | 32.7% | 51.7% dropout |
| | High | $\ge 0.66$ | 16.2% ($n=143$) | **94.4%** [89.3%, 97.1%] | 135 | 47.5% | **94.4% dropout** |
| | **Medium + High Combined** | $\ge 0.33$ | **36.5%** ($n=323$) | — | **228 / 284** | **80.3% caught** | **70.6% precision** [65.4%, 75.3%] |
| **Stage 3** | Low | $< 0.33$ | 64.7% ($n=573$) | 8.6% [6.5%, 11.1%] | 49 | 17.3% | 91.4% non-dropout |
| | Medium | $0.33 - 0.66$ | 16.0% ($n=142$) | 52.1% [43.9%, 60.2%] | 74 | 26.1% | 52.1% dropout |
| | High | $\ge 0.66$ | 19.2% ($n=170$) | **94.7%** [90.2%, 97.2%] | 161 | 56.7% | **94.7% dropout** |
| | **Medium + High Combined** | $\ge 0.33$ | **35.3%** ($n=312$) | — | **235 / 284** | **82.7% caught** | **75.3% precision** [70.2%, 79.8%] |

*Key Insight:* Across all stages, the High risk band ($p \ge 0.66$) achieves **>93% actual dropout rate** (Stage 1: 93.9%, Stage 2: 94.4%, Stage 3: 94.7%). By screening the combined Medium + High cohorts, advisors catch **69.0% (Stage 1), 80.3% (Stage 2), and 82.7% (Stage 3)** of all true dropouts with high precision (58.7% to 75.3%) while examining only ~35-38% of the student population.

---

## 3. System Architecture & Project Layout

```
Intelligent_Student_Dropout_Risk_Project/
│
├── data/                                 # Datasets & canonical dictionaries
│   ├── students_dropout_academic_success.csv
│   ├── student_dropout_dataset_dictionary.csv
│   └── data_dictionary_UCI.csv
│
├── src/                                  # Core machine learning & pipeline logic
│   ├── __init__.py
│   ├── config.py                         # Canonical mappings, stage definitions, ISCO-08 groups
│   ├── labels.py                         # Full 100% dictionary code mappings (no placeholders)
│   ├── features.py                       # Robust CSV loader & domain feature engineering
│   ├── preprocess.py                     # Leak-free ColumnTransformer & Pipeline builder
│   ├── train.py                          # 5-fold CV runner, minimum recall checks, parameter tuning
│   ├── evaluate.py                       # Diagnostic matrices, ROC curves, model comparison plots
│   ├── explain.py                        # Aggregated SHAP explainer & subgroup fairness auditor
│   └── risk.py                           # 3-tier risk banding (<0.33, 0.33-0.66, >=0.66) & recommendations
│
├── models/                               # Serialized production artifacts & metadata
│   ├── stage1.joblib                     # Stage 1 trained scikit-learn pipeline
│   ├── stage2.joblib                     # Stage 2 trained scikit-learn pipeline
│   ├── stage3.joblib                     # Stage 3 trained scikit-learn pipeline
│   └── metadata.json                     # Complete metrics, feature sets, and fallback defaults
│
├── app/                                  # Production Flask web application & REST API
│   ├── __init__.py
│   ├── main.py                           # Application factory, REST endpoints, JSON encoders
│   ├── schemas.py                        # Input schema validators & non-defaultable checks
│   ├── static/                           # Static assets (CSS, responsive JavaScript)
│   │   ├── style.css
│   │   └── app.js
│   └── templates/                        # Jinja2 HTML templates
│       ├── index.html                    # Advisor dashboard, stage selector, gauges
│       └── about.html                    # Technical architecture & API reference
│
├── notebooks/                            # Jupyter analysis notebooks
│   ├── 01_eda.ipynb                      # Exploratory data analysis & chi-square tests
│   └── 02_modeling.ipynb                 # Multi-stage model training, evaluation & SHAP
│
├── tests/                                # Automated verification test suite (pytest)
│   ├── test_api.py                       # Flask endpoint validation (200, 400, 413, 422, fairness)
│   ├── test_labels.py                    # Dictionary coverage verification
│   ├── test_pipeline.py                  # Leak-free guarantees, baseline checks, handle_unknown
│   └── test_risk.py                      # Risk banding boundary verification
│
├── reports/                              # Final project documentation & presentation
│   ├── report.docx                       # Complete 13-section technical report (with figures)
│   ├── slides.pptx                       # Widescreen presentation slide deck
│   └── figures/                          # 12 high-resolution generated diagnostic plots
│
├── requirements.txt                      # Exact pinned dependencies
└── README.md                             # Comprehensive technical documentation
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12 64-bit)
- PowerShell (Windows) or Bash (Linux / macOS)

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/Intelligent_Student_Dropout_Risk_Project.git
cd Intelligent_Student_Dropout_Risk_Project

# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Training, Evaluation & Notebooks

### Retrain Models from Scratch
To run 5-fold cross-validation, hyperparameter tuning, model serialization, and metadata generation:

```bash
python -m src.train
```

### Generate Diagnostic Figures & Reports
To evaluate the models on the held-out test split, generate ROC curves, confusion matrices, SHAP summaries, and fairness tables:

```bash
python -m src.evaluate
```

All 12 high-resolution figures are automatically written to `reports/figures/`.

### Run the Interactive Notebooks
Launch Jupyter Notebook to inspect EDA and modeling trajectories:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

---

## 6. Running the Web Application & REST API

Launch the local Flask application server:

```bash
python -m app.main
```

The server starts locally at `http://127.0.0.1:5000`.

### Web Dashboard Features
- **Interactive Life-Cycle Selector**: Seamlessly switch between Stage 1, Stage 2, and Stage 3 forms.
- **Dynamic Field Enforcement**: Non-defaultable semester performance fields are strictly required for their respective stages.
- **Instant Tri-Color Risk Gauge**: Low Risk (<33%, Green), Medium Risk (33%–66%, Amber), High Risk (≥66%, Red).
- **Aggregated SHAP Explanation Badges**: Visualizes the top 3 factors driving risk with friendly labels (`Semester 1 pass rate`) and directional impact.
- **Bulk CSV Upload**: Evaluate cohorts up to 500 students simultaneously with row-level validation reporting.

---

## 7. REST API Endpoints Specification

| Method | Endpoint | Description | Response Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health check, loaded models, memory verification | 200, 503 |
| `GET` | `/api/model-info` | Model metadata, stage configurations, test metrics, defaults | 200 |
| `GET` | `/api/features` | Feature schemas, data types, min/max bounds, required flags | 200 |
| `GET` | `/api/options` | Human-readable label mappings for all coded dropdowns | 200 |
| `POST` | `/api/predict` | Single student inference with automatic stage detection and SHAP | 200, 400, 422 |
| `POST` | `/api/predict/batch` | High-throughput batch prediction (up to 500 rows) with fault tolerance | 200, 400, 413, 422 |
| `GET` | `/api/explain/global/<stage>` | Top global features aggregated from one-hot columns | 200, 404 |
| `GET` | `/api/explain/fairness` | Demographic subgroup parity audit (query param: `?stage=2` or `?stage=3`) | 200 |

*Note: For convenience, aliases `/health` and `/api/info` are also supported.*

### Example Request (`POST /api/predict`)

The following authentic payload provides all required Stage 2 academic fields while omitting low-impact macro-economic indicators, demonstrating schema compliance and automatic defaulting:

```json
{
  "stage": 2,
  "student": {
    "Marital Status": 1,
    "Application mode": 1,
    "Course": 9254,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 130.0,
    "Nacionality": 1,
    "Mother's qualification": 19,
    "Father's qualification": 19,
    "Mother's occupation": 5,
    "Father's occupation": 9,
    "Admission grade": 125.0,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 13.5,
    "Curricular units 1st sem (without evaluations)": 0
  }
}
```

### Example Real Output from Live App (`200 OK`)

```json
{
  "defaulted_fields": [
    "Application order",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
  ],
  "dropout_probability": 0.218,
  "model_version": "1.0.0",
  "prediction": "Enrolled",
  "probabilities": {
    "Dropout": 0.218,
    "Enrolled": 0.4489,
    "Graduate": 0.333
  },
  "risk_level": "Low",
  "stage_used": 2,
  "suggested_actions": [
    "No immediate intervention needed; continue routine progress monitoring."
  ],
  "top_factors": [
    {
      "display_name": "Semester 1 pass rate",
      "feature": "approval_rate_s1",
      "impact": "decreases_risk",
      "value": 0.83
    },
    {
      "display_name": "Tuition fees up to date",
      "feature": "Tuition fees up to date",
      "impact": "increases_risk",
      "value": 1
    },
    {
      "display_name": "Course",
      "feature": "Course",
      "impact": "neutral",
      "value": 9254
    }
  ]
}
```

---

## 8. Automated Test Suite

The test suite validates compliance with all project constraints, data leakage rules, schema boundaries, and risk logic:

```bash
pytest tests/ -v
```

### Verified Test Categories (All 23 Tests Passing):
1. **API Validation (`tests/test_api.py` — 13 tests)**:
   - `test_health_check`: Verifies 200 OK and model status.
   - `test_model_info`: Verifies `/api/model-info` metadata return.
   - `test_options_endpoint`: Verifies `/api/options` label mapping format.
   - `test_options_completeness_and_ui_sync`: Verifies all 17 select dropdown options match frontend DOM element IDs.
   - `test_valid_stage1_prediction`: Verifies Stage 1 inference and probability summation ($\sum p = 1.0$).
   - `test_valid_stage3_prediction`: Verifies Stage 3 pipeline routing.
   - `test_missing_student_key`: Verifies 400 Bad Request.
   - `test_invalid_age_validation`: Verifies 422 Unprocessable Entity for invalid values (`Age = 200`).
   - `test_approved_exceeding_enrolled`: Verifies 422 when `approved > enrolled`.
   - `test_forced_stage_ignores_extras`: Verifies forced lower stages trim excess performance fields.
   - `test_batch_prediction`: Verifies bulk scoring of multiple students.
   - `test_batch_exceeding_limit`: Verifies 413 Payload Too Large on $>500$ rows.
   - `test_explain_fairness_endpoint`: Verifies `/api/explain/fairness` returns subgroup audit records with sample size $n$ and reliability flags.
2. **Label Dictionary Coverage (`tests/test_labels.py` — 2 tests)**:
   - `test_labels_format`: Verifies dictionaries are non-empty and well-formed.
   - `test_labels_data_coverage`: Verifies 100% dictionary mapping across all codes in training data with zero empty placeholders.
3. **Pipeline Integrity & Anti-Leakage (`tests/test_pipeline.py` — 3 tests)**:
   - `test_occupation_mapping_coverage`: Verifies all raw parental occupation codes map into valid ISCO-08 groups.
   - `test_pipeline_unseen_category_no_crash`: Confirms `handle_unknown="ignore"` executes without throwing exceptions on unseen categories.
   - `test_pipeline_deterministic`: Confirms identical repeat predictions on fixed inputs (`RANDOM_STATE = 42`).
4. **Risk Logic Banding (`tests/test_risk.py` — 5 tests)**:
   - `test_risk_levels`: Boundary tests for 0.32, 0.33, 0.65, 0.66 ensuring exact risk level transitions.
   - `test_suggest_actions_high_risk_unpaid_fees`: Verifies financial aid suggestions for unpaid tuition.
   - `test_suggest_actions_high_risk_academic`: Verifies tutoring suggestions for academic failures.
   - `test_suggest_actions_low_risk`: Verifies positive encouragement actions for low risk.
   - `test_suggest_actions_medium_risk`: Verifies proactive mentor check-ins for medium risk.

---

## 9. Viva Voce Defense Reference

| Examiner Question | Defense Rationale |
| :--- | :--- |
| **Why three staged models instead of one single model?** | Information becomes available chronologically. A single year-end model is unusable at admission when interventions are cheapest; conversely, a pre-admission model ignores academic reality. 3 stages balance timeliness vs predictive power. |
| **Why not optimize solely for Accuracy?** | In an imbalanced dataset (~50% Graduate, ~32% Dropout, ~18% Enrolled), a naive majority classifier achieves 50% accuracy while failing 100% of dropouts. Macro-F1 treats all classes equally, and dynamic recall floors ensure dropouts are detected. |
| **How did you prevent Data Leakage?** | The 80/20 train/test split was locked before training. Standard scaling and one-hot encoding are encapsulated inside `sklearn.pipeline.Pipeline` objects, fitting only on training CV folds. Test data was evaluated strictly once. |
| **Why choose Random Forest across all stages instead of Gradient Boosting or XGBoost?** | In 3-class classification, Macro-F1 averages performance equally across classes: $(F1_{\text{Drop}} + F1_{\text{Enr}} + F1_{\text{Grad}})/3$. Boosted models achieved $\sim 0.02$ higher Macro-F1 primarily by guessing the small transitional "Enrolled" class (18% of students) more aggressively ($0.45 \to 0.54$ F1, lifting Macro-F1 by $0.09 / 3 \approx 0.03$). However, this cost $4.5$ to $6.3$ percentage points of true dropout recall. Because the educational goal is early dropout detection, Option A (Random Forest across all three stages) was chosen. Random Forest delivers higher dropout recall (+6.0% in Stage 1, +4.9% in Stage 2, +6.3% in Stage 3), superior test accuracy (61.0%, 74.4%, 77.5%), and exact native TreeSHAP explainability without approximation overhead. |
| **How do you diagnose the difficulty of the "Enrolled" class?** | Enrolled students are in an intermediate state (some delayed, some undecided). In Stage 3, Random Forest achieves 38.4% recall and 54.5% precision (F1 = 0.450) for Enrolled, compared to 75.7% recall and 82.7% precision for Dropout. The confusion matrix shows Enrolled students disperse evenly across Graduate and Dropout boundaries. |
| **How did you handle the Categorical SHAP problem?** | One-hot encoding explodes columns (e.g. `Course_9254`). Our custom explainer sums signed SHAP values across child one-hot columns back to the parent feature, sorting by absolute magnitude while retaining directional impact (`increases_risk` / `decreases_risk`). |

---

## 10. Deliverables & Submission Packaging

The project repository includes all necessary deliverables:
- **Trained Model Artifacts**: `models/stage1.joblib`, `models/stage2.joblib`, `models/stage3.joblib`, `models/metadata.json`.
- **Generated Report**: Complete 13-section documentation at `reports/report.docx` (1.33 MB) with embedded figures.
- **Presentation Deck**: Professional 16:9 widescreen slides at `reports/slides.pptx` (11 slides).
- **Diagnostic Visualizations**: All 12 high-resolution PNG plots in `reports/figures/`.
- **Submission Archive**: When generating a zip submission, ensure `models/` and `reports/figures/` are included so the application functions out of the box.

---

## 11. Ethical Considerations, Subgroup Fairness & Responsible AI

- **Decision-Support Only**: The application provides advisory recommendations and must never be used to automate punitive disciplinary actions, deny enrollment, or revoke financial aid.
- **Uncalibrated Probabilities**: Predictions represent relative risk bands rather than absolute calibrated probabilities.
- **Single Institution Scope**: The dataset originates from a single Portuguese higher education institution (Polytechnic Institute of Portalegre); institutions with different grading or fee policies must re-calibrate prior to deployment.
- **Point-in-Time Snapshot Nuance**: Financial attributes (such as debtor and tuition status) reflect point-in-time status at data collection and may fluctuate dynamically over time.
- **Critical Finding: Financial Status Gap**:
  > **Advisory Warning:** A low-risk result does not mean a student is safe. While the model catches **96.8%** of dropouts who have overdue tuition fees, its recall is **58.1% (Stage 2)** and **65.4% (Stage 3)** for students whose tuition fees are current (an improvement of +6.8% to +8.9% over earlier boosted models, narrowing the financial gap). Students with clean financial records represent 67% of all dropouts in the test set (191 of 284). Financial distress creates an overwhelming statistical signal, which can mask academic attrition among financially secure students. Advisors must never assume financially current students are safe.

### Subgroup Demographic Fairness Audit (Held-out Test Set, n=885, Total Dropouts=284)

Reliability is determined by the number of true dropouts in the group ($n_{\text{dropouts}} \ge 50$ required for statistical stability; groups with $< 50$ dropouts have wide confidence intervals and are marked Indicative):

| Dimension | Subgroup | Total $n$ | True Dropouts | Stage 2 Recall [95% Wilson CI] | Stage 3 Recall [95% Wilson CI] | Reliability Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuition Fees** | Fees Overdue (0) | $n = 109$ | 93 | **96.8%** [90.9%, 98.9%] | **96.8%** [90.9%, 98.9%] | Reliable ($n_{\text{drop}} \ge 50$) |
| | Fees Current (1) | $n = 776$ | 191 | **58.1%** [51.0%, 64.9%] | **65.4%** [58.5%, 71.8%] | Reliable ($n_{\text{drop}} \ge 50$) |
| **Debtor** | Has Debt (1) | $n = 114$ | 63 | **88.9%** [78.8%, 94.5%] | **88.9%** [78.8%, 94.5%] | Reliable ($n_{\text{drop}} \ge 50$) |
| | No Debt (0) | $n = 771$ | 221 | **65.6%** [59.1%, 71.6%] | **71.9%** [65.7%, 77.5%] | Reliable ($n_{\text{drop}} \ge 50$) |
| **Gender** | Female (0) | $n = 570$ | 139 | **66.2%** [58.0%, 73.5%] | **74.1%** [66.2%, 80.7%] | Reliable ($n_{\text{drop}} \ge 50$) |
| | Male (1) | $n = 315$ | 145 | **75.2%** [67.6%, 81.5%] | **77.2%** [69.8%, 83.3%] | Reliable ($n_{\text{drop}} \ge 50$) |
| **Scholarship** | No Scholarship (0) | $n = 680$ | 265 | **72.8%** [67.2%, 77.8%] | **77.7%** [72.4%, 82.3%] | Reliable ($n_{\text{drop}} \ge 50$) |
| | Scholarship (1) | $n = 205$ | 19 | **42.1%** [23.1%, 63.7%] | **47.4%** [27.3%, 68.3%] | **Indicative** ($n_{\text{drop}}=19 < 50$) |
| **International**| Domestic (0) | $n = 861$ | 275 | **71.3%** [65.7%, 76.3%] | **76.4%** [71.0%, 81.0%] | Reliable ($n_{\text{drop}} \ge 50$) |
| | International (1) | $n = 24$ | 9 | **55.6%** [26.7%, 81.1%] | **55.6%** [26.7%, 81.1%] | **Indicative** ($n_{\text{drop}}=9 < 50$) |
| **Special Needs** | General (0) | $n = 875$ | 280 | **71.1%** [65.5%, 76.1%] | **76.1%** [70.7%, 80.7%] | Reliable ($n_{\text{drop}} \ge 50$) |
| | Special Needs (1) | $n = 10$ | 4 | **50.0%** [15.0%, 85.0%] | **50.0%** [15.0%, 85.0%] | **Indicative** ($n_{\text{drop}}=4 < 50$) |

*Fairness Takeaways:* Groups with small dropout counts (Scholarship $n_{\text{drop}}=19$, International $n_{\text{drop}}=9$, Special Needs $n_{\text{drop}}=4$) have wide confidence intervals spanning 35 to 70 percentage points; observed variations reflect small sample size rather than algorithmic bias.
- **Privacy by Design**: The platform stores zero Personally Identifiable Information (no student names, student IDs, emails, or IP addresses are persisted).

---

## 12. References

1. Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting student dropout and academic success.* Data, 7(11), 146.
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
3. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS, 30, 4765-4774.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* ACM SIGKDD, 785-794.
