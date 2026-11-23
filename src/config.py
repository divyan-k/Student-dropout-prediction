"""Central configuration for Smart Student Dropout Prediction System.

Contains:
- Directory and file paths
- Feature definitions by stage
- Data normalization and column mapping rules
- Model hyperparameter grids and evaluation settings
- Risk thresholds and occupation grouping logic
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

DATA_FILE = DATA_DIR / "students_dropout_academic_success.csv"
DICT_FILE = DATA_DIR / "student_dropout_dataset_dictionary.csv"
UCI_DICT_FILE = DATA_DIR / "data_dictionary_UCI.csv"

# Raw columns normalization
# Maps lowercase/variant column names to standard canonical names
CANONICAL_COLUMN_MAP = {
    "marital status": "Marital Status",
    "marital_status": "Marital Status",
    "application mode": "Application mode",
    "application_mode": "Application mode",
    "application order": "Application order",
    "application_order": "Application order",
    "course": "Course",
    "daytime/evening attendance": "Daytime/evening attendance",
    "daytime/evening attendance\t": "Daytime/evening attendance",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "previous qualification": "Previous qualification",
    "previous_qualification": "Previous qualification",
    "previous qualification (grade)": "Previous qualification (grade)",
    "previous_qualification_grade": "Previous qualification (grade)",
    "nacionality": "Nacionality",
    "nationality": "Nacionality",
    "mother's qualification": "Mother's qualification",
    "mother’s qualification": "Mother's qualification",
    "mother_qualification": "Mother's qualification",
    "father's qualification": "Father's qualification",
    "father’s qualification": "Father's qualification",
    "father_qualification": "Father's qualification",
    "mother's occupation": "Mother's occupation",
    "mother’s occupation": "Mother's occupation",
    "mother_occupation": "Mother's occupation",
    "father's occupation": "Father's occupation",
    "father’s occupation": "Father's occupation",
    "father_occupation": "Father's occupation",
    "admission grade": "Admission grade",
    "admission_grade": "Admission grade",
    "displaced": "Displaced",
    "educational special needs": "Educational special needs",
    "educational_special_needs": "Educational special needs",
    "debtor": "Debtor",
    "tuition fees up to date": "Tuition fees up to date",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "gender": "Gender",
    "scholarship holder": "Scholarship holder",
    "scholarship_holder": "Scholarship holder",
    "age at enrollment": "Age at enrollment",
    "age_at_enrollment": "Age at enrollment",
    "international": "International",
    "curricular units 1st sem (credited)": "Curricular units 1st sem (credited)",
    "curricular units 1st sem (enrolled)": "Curricular units 1st sem (enrolled)",
    "curricular units 1st sem (evaluations)": "Curricular units 1st sem (evaluations)",
    "curricular units 1st sem (approved)": "Curricular units 1st sem (approved)",
    "curricular units 1st sem (grade)": "Curricular units 1st sem (grade)",
    "curricular units 1st sem (without evaluations)": "Curricular units 1st sem (without evaluations)",
    "curricular units 2nd sem (credited)": "Curricular units 2nd sem (credited)",
    "curricular units 2nd sem (enrolled)": "Curricular units 2nd sem (enrolled)",
    "curricular units 2nd sem (evaluations)": "Curricular units 2nd sem (evaluations)",
    "curricular units 2nd sem (approved)": "Curricular units 2nd sem (approved)",
    "curricular units 2nd sem (grade)": "Curricular units 2nd sem (grade)",
    "curricular units 2nd sem (without evaluations)": "Curricular units 2nd sem (without evaluations)",
    "unemployment rate": "Unemployment rate",
    "unemployment_rate": "Unemployment rate",
    "inflation rate": "Inflation rate",
    "inflation_rate": "Inflation rate",
    "gdp": "GDP",
    "target": "target",
}

# Feature groups by type
CATEGORICAL_COLS = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

NUMERIC_COLS_BASE = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

BINARY_COLS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Stage 1: 24 raw features
STAGE1_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS_BASE + BINARY_COLS

# Stage 2 adds: 6 features from 1st semester
STAGE2_ADDS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

# Stage 3 adds: 6 features from 2nd semester
STAGE3_ADDS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# Master feature list per stage
STAGE_FEATURES = {
    1: list(STAGE1_FEATURES),
    2: list(STAGE1_FEATURES + STAGE2_ADDS),
    3: list(STAGE1_FEATURES + STAGE2_ADDS + STAGE3_ADDS),
}

# Required performance fields per stage (NEVER defaulted)
STAGE1_PERFORMANCE_FIELDS = []
STAGE2_PERFORMANCE_FIELDS = list(STAGE2_ADDS)
STAGE3_PERFORMANCE_FIELDS = list(STAGE3_ADDS)

# Fields eligible for defaults when omitted by user (strictly low-stakes context)
DEFAULTABLE_FIELDS = [
    "Application order",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Displaced",
    "Educational special needs",
    "International",
    "Marital Status",
    "Nacionality",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Daytime/evening attendance",
]

# Target definitions
TARGET_COL = "target"
TARGET_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

# Validation ranges
RANGES = {
    "Age at enrollment": (15, 80),
    "Admission grade": (0.0, 200.0),
    "Previous qualification (grade)": (0.0, 200.0),
    "Curricular units 1st sem (grade)": (0.0, 20.0),
    "Curricular units 2nd sem (grade)": (0.0, 20.0),
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.33,
    "high": 0.66,
}

# Random state & cross validation
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Occupation grouping logic based on Portuguese Classification of Occupations / ISCO-08
def map_occupation(code: int) -> str:
    """Map Portuguese occupation code to major group: white_collar, blue_collar, or other."""
    try:
        code_int = int(code)
    except (ValueError, TypeError):
        return "other"

    if code_int in (0, 10, 90, 99):
        return "other"

    code_str = str(code_int).strip()
    if not code_str:
        return "other"

    leading = int(code_str[0])
    if leading in {1, 2, 3, 4}:
        return "white_collar"
    elif leading in {5, 6, 7, 8, 9}:
        return "blue_collar"
    return "other"

