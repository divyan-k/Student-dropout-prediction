# Smart Student Dropout Prediction System

A staged machine-learning system that predicts whether a student will **Drop out**, stay **Enrolled**, or **Graduate**, and explains why. It is delivered as a Flask web application with a REST API.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Folder Structure](#5-folder-structure)
6. [Environment Setup](#6-environment-setup)
7. [ML Pipeline](#7-ml-pipeline)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Explainability and Fairness](#9-explainability-and-fairness)
10. [Risk Logic](#10-risk-logic)
11. [REST API Specification](#11-rest-api-specification)
12. [Web UI Specification](#12-web-ui-specification)
13. [Error Handling and Validation](#13-error-handling-and-validation)
14. [Testing Plan](#14-testing-plan)
15. [Day-wise Build Plan](#15-day-wise-build-plan)
16. [Report Outline](#16-report-outline)
17. [Viva Preparation](#17-viva-preparation)
18. [Limitations and Ethics](#18-limitations-and-ethics)
19. [Future Scope](#19-future-scope)
20. [Definition of Done](#20-definition-of-done)
21. [References](#21-references)

---

## 1. Project Overview

**Problem.** Student dropout harms students and institutions. Staff usually notice at-risk students too late. This project builds a decision-support tool that flags risk early and explains the reasons.

**Solution.** A web app where a staff member enters a student's details and receives:

- Predicted outcome: Dropout / Enrolled / Graduate
- Probability for each outcome
- Risk level: Low / Medium / High
- Top factors behind the prediction
- A suggested action

**Key idea: staged prediction.** Three models, each using the data available at a different point in the student's journey:

| Stage | Name | Data used | When it is used |
|---|---|---|---|
| 1 | Admission | Background, finance, application data | At enrollment |
| 2 | After Semester 1 | Stage 1 + 1st semester performance | After first semester results |
| 3 | After Semester 2 | Stage 2 + 2nd semester performance | After second semester results |

Early feasibility tests (untuned, single split) on the dataset showed:

| Stage | Approx. accuracy | Approx. dropout recall |
|---|---|---|
| 1 | 64% | 62% |
| 2 | 73% | 72% |
| 3 | 77% | 77% |

These are indicative only. Final numbers come from tuned, cross-validated models.

---

## 2. Objectives

1. Analyse factors associated with student dropout.
2. Build and compare machine-learning models for 3-class outcome prediction.
3. Design three staged models for early, mid and late intervention.
4. Explain predictions using feature importance and SHAP.
5. Deploy the models behind a REST API and a simple web interface.
6. Document limitations, fairness and ethics honestly.

---

## 3. Dataset

- **Source:** *Predict Students' Dropout and Academic Success* (Realinho et al., 2021), Polytechnic Institute of Portalegre, Portugal.
- **Files:**
  - `students_dropout_academic_success.csv` (data)
  - `student_dropout_dataset_dictionary.csv` (feature descriptions)
- **Size:** 4,424 rows, 36 features, 1 target column named `target`.
- **Quality:** no missing values, no duplicate rows.

### 3.1 Target distribution

| Class | Count | Share |
|---|---|---|
| Graduate | 2,209 | 50% |
| Dropout | 1,421 | 32% |
| Enrolled | 794 | 18% |

The classes are imbalanced, so accuracy alone is misleading. Use dropout recall and macro-F1.

### 3.2 Feature groups

**Stage 1: Admission-time features (24)**

| Type | Columns |
|---|---|
| Categorical codes | `Marital Status`, `Application mode`, `Course`, `Previous qualification`, `Nacionality`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation` |
| Numeric | `Application order`, `Previous qualification (grade)`, `Admission grade`, `Age at enrollment`, `Unemployment rate`, `Inflation rate`, `GDP` |
| Binary | `Daytime/evening attendance`, `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `International` |

**Stage 2 adds (6):**
`Curricular units 1st sem (credited)`, `(enrolled)`, `(evaluations)`, `(approved)`, `(grade)`, `(without evaluations)`

**Stage 3 adds (6):**
`Curricular units 2nd sem (credited)`, `(enrolled)`, `(evaluations)`, `(approved)`, `(grade)`, `(without evaluations)`

> Note: the column is spelled `Nacionality` in the file. Use exactly this spelling in code.

### 3.3 Engineered features

| Feature | Formula | Stage |
|---|---|---|
| `approval_rate_s1` | approved ÷ enrolled (0 if enrolled = 0) | 2, 3 |
| `approval_rate_s2` | approved ÷ enrolled (0 if enrolled = 0) | 3 |
| `grade_change` | 2nd sem grade − 1st sem grade | 3 |
| `total_approved` | 1st + 2nd sem approved | 3 |
| `parent_occupation_group` | Group occupation codes into white-collar / blue-collar / other | 1, 2, 3 |

### 3.4 Target encoding

| Label | Code |
|---|---|
| Dropout | 0 |
| Enrolled | 1 |
| Graduate | 2 |

---

## 4. System Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────────┐
│  Web Form    │ ──► │ Flask API     │ ──► │ Validation + Stage   │
│ (HTML/JS)    │     │ /api/predict  │     │ Selector             │
└──────────────┘     └───────────────┘     └──────────┬───────────┘
        ▲                                             │
        │                                             ▼
        │            ┌────────────────┐     ┌──────────────────────┐
        └─────────── │ JSON Response  │ ◄── │ sklearn Pipeline     │
                     │ + risk + SHAP  │     │ (preprocess + model) │
                     └────────────────┘     └──────────────────────┘
```

**Training flow (offline):**

```
CSV → clean → split (stratified) → per-stage Pipeline
    → 5-fold CV comparison → tune best → test once → save .joblib
```

**Design principle:** each saved file is a complete sklearn `Pipeline` (preprocessing + classifier). This guarantees that training and serving apply identical transformations.

---

## 5. Folder Structure

```
dropout-prediction/
├── data/
│   ├── students_dropout_academic_success.csv
│   └── student_dropout_dataset_dictionary.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── config.py            # column groups, stage definitions, paths
│   ├── features.py          # engineered features
│   ├── preprocess.py        # builds the ColumnTransformer per stage
│   ├── train.py             # ModelRunner class: train / evaluate / tune
│   ├── evaluate.py          # metrics and plots
│   ├── explain.py           # feature importance and SHAP helpers
│   └── risk.py              # probability to risk level and suggestions
├── models/
│   ├── stage1.joblib
│   ├── stage2.joblib
│   ├── stage3.joblib
│   └── metadata.json        # metrics, feature lists, defaults, version
├── app/
│   ├── main.py              # Flask app and routes
│   ├── schemas.py           # request validation
│   ├── templates/
│   │   ├── index.html
│   │   └── about.html
│   └── static/
│       ├── style.css
│       └── app.js
├── tests/
│   ├── test_api.py
│   └── test_risk.py
├── reports/
│   ├── figures/
│   └── report.docx
├── requirements.txt
└── README.md
```

---

## 6. Environment Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
scipy
joblib
flask
flask-cors
pytest
jupyter
```

**Run training, then the app:**

```bash
python -m src.train          # trains all three stages, saves models/
python -m app.main           # starts server at http://127.0.0.1:5000
```

---

## 7. ML Pipeline

### 7.1 Preprocessing

| Column type | Transform |
|---|---|
| Category codes (9 columns) | `OneHotEncoder(handle_unknown="ignore")` |
| Numeric columns | `StandardScaler` |
| Binary columns | Passed through unchanged |

All transforms live inside a `ColumnTransformer` inside a `Pipeline`. They are fitted on training data only.

### 7.2 Split

- Stratified 80/20 train/test split, `random_state=42`.
- The split happens **before** any fitting.
- The test set is used once, at the end.

### 7.3 Imbalance handling

1. **Primary:** `class_weight="balanced"` where supported.
2. **Experiment:** SMOTENC (handles categorical columns), applied inside the training folds only.
3. Report which approach was kept and why.

### 7.4 Models compared (per stage)

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Simple, visual |
| Random Forest | Strong ensemble |
| Gradient Boosting / XGBoost | Strong boosted ensemble |
| SVM (optional) | Extra comparison |

### 7.5 Tuning

`RandomizedSearchCV` with 5-fold stratified CV, scoring on macro-F1 (or dropout recall if that is chosen as the headline).

Example search space for Random Forest:

```python
{
  "n_estimators": [200, 300, 500],
  "max_depth": [None, 8, 12, 16],
  "min_samples_leaf": [1, 2, 4],
  "max_features": ["sqrt", "log2"],
}
```

### 7.6 ModelRunner class (src/train.py)

```python
class ModelRunner:
    def __init__(self, stage: int, model, name: str): ...
    def train(self, X_train, y_train): ...        # fits the pipeline
    def cross_validate(self, X, y, cv=5): ...      # returns mean/std of metrics
    def tune(self, X, y, param_grid, n_iter=20): ...
    def evaluate(self, X_test, y_test): ...        # accuracy, macro-F1, per-class report
    def save(self, path): ...                      # joblib.dump
```

### 7.7 metadata.json (saved with models)

```json
{
  "version": "1.0.0",
  "trained_on": "YYYY-MM-DD",
  "classes": ["Dropout", "Enrolled", "Graduate"],
  "stages": {
    "1": {"features": ["..."], "model": "RandomForest", "metrics": {"accuracy": 0, "macro_f1": 0, "dropout_recall": 0}},
    "2": {"features": ["..."], "model": "GradientBoosting", "metrics": {}},
    "3": {"features": ["..."], "model": "RandomForest", "metrics": {}}
  },
  "defaults": {"Application order": 1, "Admission grade": 126.0}
}
```

`defaults` holds training-set medians (numeric) and modes (categorical) for any field the user does not supply.

---

## 8. Evaluation Strategy

| Metric | Why |
|---|---|
| **Dropout recall** | Missing a real dropout is the costly error |
| **Macro-F1** | Treats all three classes equally, exposes the weak Enrolled class |
| Accuracy | Context only |
| Per-class precision/recall/F1 | Full picture |
| Confusion matrix | Shows where errors occur |
| ROC-AUC (one-vs-rest) | Threshold-independent quality |

**Baselines to beat:** majority-class prediction (~50% accuracy).

**Protocol:**

1. Compare all models with 5-fold CV on the training set only.
2. Tune the best per stage.
3. Evaluate on the test set once.
4. Report CV mean ± std alongside test results.

**Required result table in the report:**

| Stage | Model | Accuracy | Macro-F1 | Dropout recall | ROC-AUC |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

**Known weakness to report honestly:** the Enrolled class is hardest to predict, because those students are a mix of future graduates and future dropouts.

---

## 9. Explainability and Fairness

### 9.1 Explainability

- Global: tree feature importances and a SHAP summary plot per stage.
- Local: SHAP values for a single prediction, returning the top 3 factors pushing toward or away from dropout.
- Output format used by the API:

```json
"top_factors": [
  {"feature": "Curricular units 2nd sem (approved)", "value": 1, "impact": "increases_risk"},
  {"feature": "Tuition fees up to date", "value": 0, "impact": "increases_risk"},
  {"feature": "Scholarship holder", "value": 1, "impact": "decreases_risk"}
]
```

### 9.2 Fairness checks

- Compare dropout recall across `Gender`, `Scholarship holder`, `International` and age groups.
- Document any gap above ~5 percentage points.
- Discuss whether sensitive attributes (gender, nationality, marital status) should be excluded, and test the effect of removing them.

---

## 10. Risk Logic

Risk is based on the predicted **dropout probability** `p`.

| Risk level | Condition |
|---|---|
| Low | p < 0.33 |
| Medium | 0.33 ≤ p < 0.66 |
| High | p ≥ 0.66 |

Thresholds are adjustable in `src/risk.py`. Choose final values using validation data and the recall target.

**Suggested actions (rule-based):**

| Condition | Suggestion |
|---|---|
| High risk and `Tuition fees up to date` = 0 | Refer to financial aid and fee-support office |
| High risk and low approval rate | Arrange academic counselling and tutoring |
| Medium risk | Schedule a mentor check-in |
| High risk and `Debtor` = 1 | Review payment plan options |
| Low risk | No action; continue routine monitoring |

---

## 11. REST API Specification

**Base URL:** `http://127.0.0.1:5000`
**Content type:** `application/json`
**Authentication:** none (course project). See [Future Scope](#19-future-scope).

### Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve the web form |
| GET | `/about` | Serve the project and model info page |
| GET | `/api/health` | Service health check |
| GET | `/api/model-info` | Model versions, metrics, class labels |
| GET | `/api/features` | Feature list per stage with types and allowed values |
| GET | `/api/options` | Dropdown choices (readable labels) for categorical fields |
| POST | `/api/predict` | Predict a single student |
| POST | `/api/predict/batch` | Predict many students |
| GET | `/api/explain/global/<stage>` | Global feature importance for a stage |

---

### 11.1 `GET /api/health`

**Response 200**

```json
{
  "status": "ok",
  "models_loaded": [1, 2, 3],
  "version": "1.0.0"
}
```

---

### 11.2 `GET /api/model-info`

**Response 200**

```json
{
  "version": "1.0.0",
  "classes": ["Dropout", "Enrolled", "Graduate"],
  "stages": {
    "1": {"name": "Admission", "model": "RandomForest", "metrics": {"accuracy": 0.64, "macro_f1": 0.55, "dropout_recall": 0.62}},
    "2": {"name": "After Semester 1", "model": "GradientBoosting", "metrics": {}},
    "3": {"name": "After Semester 2", "model": "RandomForest", "metrics": {}}
  }
}
```

> Values above are placeholders. The real values come from `models/metadata.json`.

---

### 11.3 `GET /api/features`

**Query params:** `stage` (optional, 1 to 3)

**Response 200**

```json
{
  "stage": 2,
  "features": [
    {"name": "Admission grade", "type": "numeric", "min": 0, "max": 200, "required": true},
    {"name": "Debtor", "type": "binary", "allowed": [0, 1], "required": true},
    {"name": "Course", "type": "category", "allowed_source": "/api/options", "required": true}
  ]
}
```

---

### 11.4 `GET /api/options`

Returns human-readable labels for coded fields so the UI can show dropdowns instead of raw codes.

**Response 200**

```json
{
  "Course": [{"code": 33, "label": "Biofuel Production Technologies"}],
  "Marital Status": [{"code": 1, "label": "Single"}],
  "Application mode": [{"code": 1, "label": "1st phase - general contingent"}],
  "Gender": [{"code": 0, "label": "Female"}, {"code": 1, "label": "Male"}]
}
```

> Build this mapping from `student_dropout_dataset_dictionary.csv` and the dataset's documented code lists.

---

### 11.5 `POST /api/predict`

Predicts one student. The stage is chosen automatically from the fields supplied, or can be forced with `stage`.

**Request body**

```json
{
  "stage": 2,
  "student": {
    "Marital Status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 9254,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 130.0,
    "Nacionality": 1,
    "Mother's qualification": 19,
    "Father's qualification": 12,
    "Mother's occupation": 5,
    "Father's occupation": 9,
    "Admission grade": 125.0,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 19,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 3,
    "Curricular units 1st sem (grade)": 11.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74
  }
}
```

**Rules**

- `student` is required. `stage` is optional (1, 2 or 3).
- If `stage` is omitted: Stage 3 if any 2nd-semester field is present, else Stage 2 if any 1st-semester field is present, else Stage 1.
- Missing optional fields are filled from `defaults` in `metadata.json`, and the response lists which were defaulted.

**Response 200**

```json
{
  "stage_used": 2,
  "prediction": "Dropout",
  "probabilities": {"Dropout": 0.58, "Enrolled": 0.17, "Graduate": 0.25},
  "dropout_probability": 0.58,
  "risk_level": "Medium",
  "suggested_actions": ["Schedule a mentor check-in"],
  "top_factors": [
    {"feature": "Curricular units 1st sem (approved)", "value": 3, "impact": "increases_risk"},
    {"feature": "Curricular units 1st sem (grade)", "value": 11.5, "impact": "increases_risk"},
    {"feature": "Tuition fees up to date", "value": 1, "impact": "decreases_risk"}
  ],
  "defaulted_fields": ["Application order"],
  "model_version": "1.0.0"
}
```

**Error responses:** see [Section 13](#13-error-handling-and-validation).

---

### 11.6 `POST /api/predict/batch`

**Request body**

```json
{
  "stage": 3,
  "students": [ { "...": "same fields as /api/predict" }, { "...": "..." } ]
}
```

**Limits:** maximum 500 students per request.

**Response 200**

```json
{
  "stage_used": 3,
  "count": 2,
  "results": [
    {"index": 0, "prediction": "Graduate", "dropout_probability": 0.07, "risk_level": "Low"},
    {"index": 1, "prediction": "Dropout", "dropout_probability": 0.81, "risk_level": "High"}
  ],
  "errors": []
}
```

A single invalid row does not fail the whole batch: it appears in `errors` with its index and message.

---

### 11.7 `GET /api/explain/global/<stage>`

**Path param:** `stage` is 1, 2 or 3.

**Response 200**

```json
{
  "stage": 3,
  "method": "tree_feature_importance",
  "top_features": [
    {"feature": "Curricular units 2nd sem (approved)", "importance": 0.14},
    {"feature": "Curricular units 2nd sem (grade)", "importance": 0.11}
  ]
}
```

---

### 11.8 Example calls

```bash
curl http://127.0.0.1:5000/api/health

curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @sample_student.json
```

```python
import requests
r = requests.post("http://127.0.0.1:5000/api/predict", json={"stage": 1, "student": {...}})
print(r.json()["risk_level"])
```

---

## 12. Web UI Specification

### 12.1 Pages

| Page | Content |
|---|---|
| `/` | Prediction form and results |
| `/about` | Project description, model metrics, limitations, ethics note |

### 12.2 Form layout

Three collapsible sections that unlock the stages:

1. **Admission details** (always shown): course, application mode, previous qualification, admission grade, age, gender, scholarship, debtor, tuition up to date, parents' details.
2. **Semester 1 results** (optional): enrolled, approved, evaluations, grade.
3. **Semester 2 results** (optional): enrolled, approved, evaluations, grade.

A badge shows which stage will be used based on the filled sections.

### 12.3 UX rules

- Dropdowns with readable labels from `/api/options`.
- Only the most important inputs are shown in a main view; the rest sit under an "Advanced fields" toggle with defaults filled in.
- Client-side checks before submitting (see Section 13).
- Loading indicator during prediction.

### 12.4 Result panel

- Predicted outcome in large text.
- Coloured risk badge: green (Low), amber (Medium), red (High).
- Probability bars for the three classes.
- "Why?" list with the top 3 factors.
- Suggested actions.
- Note: "Decision-support only. Not a final judgement about a student."

---

## 13. Error Handling and Validation

### 13.1 Field validation rules

| Field | Rule |
|---|---|
| Binary fields | Must be 0 or 1 |
| `Age at enrollment` | 15 to 80 |
| `Admission grade`, `Previous qualification (grade)` | 0 to 200 |
| Semester grade | 0 to 20 |
| Units fields | Non-negative integers |
| Approved units | Must be ≤ enrolled units (+ credited if used) |
| `Course`, `Application mode` etc. | Must be a known code |

### 13.2 Error response format

```json
{
  "error": "validation_error",
  "message": "One or more fields are invalid.",
  "details": [
    {"field": "Age at enrollment", "problem": "must be between 15 and 80"}
  ]
}
```

### 13.3 Status codes

| Code | When |
|---|---|
| 200 | Success |
| 400 | Malformed JSON or missing `student` |
| 404 | Unknown route or stage |
| 413 | Batch above 500 rows |
| 422 | Field validation failed |
| 500 | Unexpected server error (logged, generic message returned) |
| 503 | Models not loaded |

---

## 14. Testing Plan

### 14.1 API tests (`tests/test_api.py`)

| Test | Expectation |
|---|---|
| Health check | 200, `status: ok` |
| Valid Stage 1 payload | 200, valid probabilities summing to ~1 |
| Valid Stage 3 payload | 200, `stage_used = 3` |
| Missing `student` key | 400 |
| Age = 200 | 422 with field detail |
| Approved > enrolled | 422 |
| Unknown course code | 422 |
| Batch with one bad row | 200, error listed for that row |
| Batch of 501 rows | 413 |

### 14.2 Risk-logic tests (`tests/test_risk.py`)

Boundary values 0.32, 0.33, 0.65, 0.66 map to the correct levels.

### 14.3 Manual scenarios

| Scenario | Expected risk |
|---|---|
| Unpaid fees, debtor, 0 approved units | High |
| Scholarship, many approved units, high grades | Low |
| Mixed signals | Medium |
| Only admission data supplied | Stage 1 used |

### 14.4 Model sanity checks

- Predictions are identical for the same input run twice.
- Unseen category code does not crash the pipeline.

---

## 15. Day-wise Build Plan

| Day | Focus | Deliverables |
|---|---|---|
| 1 | Setup and data audit | Folder structure, environment, feature groups defined in `config.py`, problem statement |
| 2 | EDA and statistical tests | 8 to 10 figures, chi-square tests, 6 written findings |
| 3 | Preprocessing and feature engineering | `preprocess.py`, `features.py`, leak-free pipeline per stage |
| 4 | Model comparison | `ModelRunner`, CV comparison table per stage, imbalance experiment |
| 5 | Tuning and final evaluation | Tuned models, test-set metrics, confusion matrices, saved `.joblib` files, `metadata.json` |
| 6 | Explainability, fairness, risk | SHAP plots, fairness table, `risk.py` |
| 7 | Flask app | All API endpoints and the web form working for all three stages |
| 8 | Testing and report | Passing tests, README, report draft |
| 9 | Polish and viva prep | Slides, demo video, fresh-install check, viva rehearsal |

**If time runs short, cut in this order:** SMOTENC, SVM, batch endpoint, `/about` page. Keep the three stages, SHAP and the working app.

---

## 16. Report Outline

1. Introduction and problem statement
2. Objectives
3. Related work (cite Realinho et al., 2021, and any reference projects)
4. Dataset description
5. Methodology: preprocessing, feature engineering, staged design, metrics
6. EDA and statistical tests
7. Results: model comparison and the three-stage table
8. Explainability and fairness
9. System design and implementation, with architecture diagram, API table and screenshots
10. Testing
11. Limitations and ethics
12. Conclusion and future work
13. References

**Figures to include:** target distribution, dropout rate by key factors, correlation heatmap, model comparison chart, confusion matrices, ROC curves, SHAP summary, architecture diagram, app screenshots.

---

## 17. Viva Preparation

| Question | Key points for the answer |
|---|---|
| Why three staged models? | Information becomes available over time; earlier prediction allows earlier intervention at some cost in accuracy |
| Why not only accuracy? | Imbalanced classes; dropout recall and macro-F1 show real performance |
| How did you avoid data leakage? | Split first, all fitting inside a Pipeline, test set used once |
| How did you handle imbalance? | Class weights, with SMOTENC compared inside CV folds |
| Why that final model? | State the CV evidence and the metric priority |
| How do you know it is not overfitting? | CV mean ± std, test score close to CV score |
| What are the most important features? | Show SHAP and explain they make sense (approved units, tuition status) |
| Why is Enrolled hard to predict? | These students have not yet reached a final outcome |
| What are the ethical risks? | Labelling students, bias, over-reliance; tool is decision support only |
| How could this be deployed? | Docker, authentication, database, monitoring and retraining |
| Why save a whole Pipeline? | Guarantees the same preprocessing at training and serving |
| What does `handle_unknown="ignore"` do? | Prevents crashes on unseen category codes |

---

## 18. Limitations and Ethics

**Limitations**

- Data from a single institution in Portugal, so results may not generalise elsewhere.
- Class imbalance and a weak Enrolled class.
- Some features (for example tuition status, debtor) are snapshots and may change over time.
- Correlation is not causation: the model shows association, not reasons.
- No temporal validation (the data is not split by cohort year).

**Ethics**

- The tool supports human decisions and must not be used to penalise or exclude students.
- Sensitive attributes (gender, nationality, marital status) carry bias risk. Document fairness checks and consider excluding them.
- Predictions can become self-fulfilling if students are labelled "high risk".
- Student data requires privacy protection: store no personal identifiers.

---

## 19. Future Scope

- Authentication and role-based access.
- Database to store predictions and outcomes for monitoring.
- Docker container and cloud deployment.
- Model monitoring and scheduled retraining.
- Temporal validation across cohort years.
- Calibrated probabilities for more reliable risk levels.
- Counterfactual explanations ("what would reduce this student's risk?").
- Dashboard for institution-level trends.

---

## 20. Definition of Done

- [ ] Three trained and saved pipelines with documented metrics
- [ ] `metadata.json` with features, metrics and defaults
- [ ] All API endpoints implemented and tested
- [ ] Web form works for all three stages
- [ ] SHAP plots and a fairness note
- [ ] Tests pass (`pytest`)
- [ ] README with setup, training and run steps
- [ ] Report, slides and demo video complete
- [ ] Project runs from a clean folder using `requirements.txt`

---

## 21. References

1. Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting student dropout and academic success.* Data, 7(11), 146. (Dataset: UCI Machine Learning Repository, "Predict Students' Dropout and Academic Success".)
2. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
3. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
4. Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR 16.
5. Flask documentation: https://flask.palletsprojects.com
