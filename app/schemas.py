"""Request and response schemas and validation logic for Flask API.

Implements rules from PROJECT_SPEC Section 13:
- Binary fields: must be 0 or 1
- Age at enrollment: 15 to 80
- Admission grade, Previous qualification grade: 0 to 200
- Semester grades: 0 to 20
- Unit counts: non-negative integers
- Approved units <= enrolled (+ credited if applicable)
- Categorical values checked against known VALID_CODES
- Non-defaultable semester performance fields checked per stage
- Automatic stage detection and forced stage trimming
"""

from typing import Dict, Any, List, Tuple
from src.config import (
    RANGES,
    DEFAULTABLE_FIELDS,
    STAGE1_FEATURES,
    STAGE2_PERFORMANCE_FIELDS,
    STAGE3_PERFORMANCE_FIELDS,
    BINARY_COLS,
    CATEGORICAL_COLS,
)
from src.labels import VALID_CODES


def detect_stage(student_data: Dict[str, Any]) -> int:
    """Infer prediction stage based on presence of semester performance fields."""
    has_s2 = any(
        k in student_data and student_data[k] is not None
        for k in STAGE3_PERFORMANCE_FIELDS
    )
    if has_s2:
        return 3

    has_s1 = any(
        k in student_data and student_data[k] is not None
        for k in STAGE2_PERFORMANCE_FIELDS
    )
    if has_s1:
        return 2

    return 1


def handle_forced_stage(student_data: Dict[str, Any], forced_stage: int) -> Tuple[Dict[str, Any], List[str]]:
    """Trim extra fields if a lower stage was explicitly forced."""
    trimmed = dict(student_data)
    ignored_fields = []

    if forced_stage == 1:
        for f in STAGE2_PERFORMANCE_FIELDS + STAGE3_PERFORMANCE_FIELDS:
            if f in trimmed:
                ignored_fields.append(f)
                del trimmed[f]
    elif forced_stage == 2:
        for f in STAGE3_PERFORMANCE_FIELDS:
            if f in trimmed:
                ignored_fields.append(f)
                del trimmed[f]

    return trimmed, ignored_fields


def fill_defaults(
    student_data: Dict[str, Any],
    stage: int,
    defaults_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Fill missing optional admission fields with defaults. Never defaults performance fields."""
    filled = dict(student_data)
    defaulted_fields = []

    for field in DEFAULTABLE_FIELDS:
        if (field not in filled or filled[field] is None) and field in defaults_dict:
            filled[field] = defaults_dict[field]
            defaulted_fields.append(field)

    return filled, defaulted_fields


def validate_student(student_data: Dict[str, Any], stage: int) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Validate student record against all schema rules."""
    cleaned = dict(student_data)
    errors: List[Dict[str, str]] = []

    # 1. Check required performance fields for this stage
    if stage >= 2:
        for f in [
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (grade)",
        ]:
            if f not in cleaned or cleaned[f] is None:
                errors.append({"field": f, "problem": f"Required for Stage {stage} prediction."})
    if stage >= 3:
        for f in [
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (grade)",
        ]:
            if f not in cleaned or cleaned[f] is None:
                errors.append({"field": f, "problem": f"Required for Stage {stage} prediction."})

    # 2. Check binary fields
    for b_col in BINARY_COLS:
        if b_col in cleaned and cleaned[b_col] is not None:
            try:
                val = int(cleaned[b_col])
                if val not in (0, 1):
                    errors.append({"field": b_col, "problem": "Must be 0 or 1."})
                cleaned[b_col] = val
            except (ValueError, TypeError):
                errors.append({"field": b_col, "problem": "Must be a binary integer (0 or 1)."})

    # 3. Numeric range validations
    for col, (min_v, max_v) in RANGES.items():
        if col in cleaned and cleaned[col] is not None:
            try:
                val = float(cleaned[col])
                if val < min_v or val > max_v:
                    errors.append({
                        "field": col,
                        "problem": f"Value {val} outside allowed range [{min_v}, {max_v}].",
                    })
                cleaned[col] = val
            except (ValueError, TypeError):
                errors.append({"field": col, "problem": "Must be a numeric value."})

    # 4. Units integer and logic checks
    unit_cols = [c for c in cleaned if "Curricular units" in c and "grade" not in c]
    for uc in unit_cols:
        try:
            val = int(cleaned[uc])
            if val < 0:
                errors.append({"field": uc, "problem": "Units count cannot be negative."})
            cleaned[uc] = val
        except (ValueError, TypeError):
            errors.append({"field": uc, "problem": "Must be a non-negative integer."})

    # Approved units <= enrolled + credited
    if stage >= 2:
        enr1 = cleaned.get("Curricular units 1st sem (enrolled)", 0)
        app1 = cleaned.get("Curricular units 1st sem (approved)", 0)
        crd1 = cleaned.get("Curricular units 1st sem (credited)", 0)
        if app1 > (enr1 + crd1):
            errors.append({
                "field": "Curricular units 1st sem (approved)",
                "problem": "Approved units cannot exceed enrolled + credited units.",
            })

    if stage >= 3:
        enr2 = cleaned.get("Curricular units 2nd sem (enrolled)", 0)
        app2 = cleaned.get("Curricular units 2nd sem (approved)", 0)
        crd2 = cleaned.get("Curricular units 2nd sem (credited)", 0)
        if app2 > (enr2 + crd2):
            errors.append({
                "field": "Curricular units 2nd sem (approved)",
                "problem": "Approved units cannot exceed enrolled + credited units.",
            })

    # 5. Categorical valid codes check
    for cat_col in CATEGORICAL_COLS:
        if cat_col in cleaned and cleaned[cat_col] is not None:
            try:
                code_val = int(cleaned[cat_col])
                cleaned[cat_col] = code_val
                if cat_col in VALID_CODES and code_val not in VALID_CODES[cat_col]:
                    errors.append({
                        "field": cat_col,
                        "problem": f"Invalid or unrecognized code {code_val}.",
                    })
            except (ValueError, TypeError):
                errors.append({"field": cat_col, "problem": "Must be an integer code."})

    return cleaned, errors

