"""Data loading and feature engineering pipelines.

Includes:
- Robust loader supporting both semicolon and comma delimited CSVs
- Automatic whitespace stripping and curly quote normalization
- Case-insensitive canonical column mapping
- Engineered metrics (approval rates, zero enrollments, grade dynamics)
- ISCO-08 parent occupation grouping
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from src.config import (
    DATA_FILE,
    CANONICAL_COLUMN_MAP,
    STAGE_FEATURES,
    TARGET_COL,
    TARGET_MAP,
    map_occupation,
)


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Robustly load student dropout dataset.
    
    Handles:
    - Comma or semicolon delimiters
    - Trailing whitespace/tabs in column names
    - Unicode curly apostrophes (' vs ')
    - Case variations in column names
    - Target mapping to integer codes (Dropout: 0, Enrolled: 1, Graduate: 2)
    """
    path = file_path or str(DATA_FILE)

    # 1. Try semicolon first, then comma
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=",")

    # 2. Clean column names
    clean_cols = {}
    for col in df.columns:
        norm = col.strip().replace("’", "'").replace("\t", "")
        norm_lower = norm.lower()
        if norm_lower in CANONICAL_COLUMN_MAP:
            clean_cols[col] = CANONICAL_COLUMN_MAP[norm_lower]
        else:
            clean_cols[col] = norm

    df = df.rename(columns=clean_cols)

    # 3. Clean string values in target if present
    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == object or isinstance(df[TARGET_COL].iloc[0], str):
            df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
            df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)

    return df


def engineer_features(df: pd.DataFrame, stage: int = 3) -> pd.DataFrame:
    """Engineer stage-appropriate features.

    Features engineered:
    - All stages:
        - mother_occupation_group, father_occupation_group
    - Stage >= 2:
        - approval_rate_s1 = approved / enrolled (0 if enrolled == 0)
        - enrolled_zero_s1 = 1 if enrolled == 0, else 0
    - Stage >= 3:
        - approval_rate_s2 = approved / enrolled (0 if enrolled == 0)
        - enrolled_zero_s2 = 1 if enrolled == 0, else 0
        - grade_change = 2nd sem grade - 1st sem grade
        - total_approved = 1st sem approved + 2nd sem approved
    """
    df_out = df.copy()

    # All stages: Occupation group
    if "Mother's occupation" in df_out.columns:
        df_out["mother_occupation_group"] = df_out["Mother's occupation"].apply(map_occupation)
    if "Father's occupation" in df_out.columns:
        df_out["father_occupation_group"] = df_out["Father's occupation"].apply(map_occupation)

    # Drop raw mother/father occupation columns to avoid redundancy
    cols_to_drop = [c for c in ["Mother's occupation", "Father's occupation"] if c in df_out.columns]
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)

    # Stage 2 features
    if stage >= 2:
        enrolled_s1 = df_out.get("Curricular units 1st sem (enrolled)", 0)
        approved_s1 = df_out.get("Curricular units 1st sem (approved)", 0)

        df_out["enrolled_zero_s1"] = (enrolled_s1 == 0).astype(int)
        df_out["approval_rate_s1"] = np.where(enrolled_s1 > 0, approved_s1 / enrolled_s1, 0.0)

    # Stage 3 features
    if stage >= 3:
        enrolled_s2 = df_out.get("Curricular units 2nd sem (enrolled)", 0)
        approved_s2 = df_out.get("Curricular units 2nd sem (approved)", 0)
        grade_s1 = df_out.get("Curricular units 1st sem (grade)", 0.0)
        grade_s2 = df_out.get("Curricular units 2nd sem (grade)", 0.0)

        df_out["enrolled_zero_s2"] = (enrolled_s2 == 0).astype(int)
        df_out["approval_rate_s2"] = np.where(enrolled_s2 > 0, approved_s2 / enrolled_s2, 0.0)
        df_out["grade_change"] = grade_s2 - grade_s1
        df_out["total_approved"] = df_out.get("Curricular units 1st sem (approved)", 0) + approved_s2

    return df_out


def get_feature_columns(stage: int) -> List[str]:
    """Return final ordered list of feature column names for a given stage."""
    raw_cols = list(STAGE_FEATURES[stage])

    # Remove raw occupations that were replaced
    cols = [c for c in raw_cols if c not in ["Mother's occupation", "Father's occupation"]]

    # Add engineered occupation categories
    cols.extend(["mother_occupation_group", "father_occupation_group"])

    if stage >= 2:
        cols.extend(["enrolled_zero_s1", "approval_rate_s1"])

    if stage >= 3:
        cols.extend(["enrolled_zero_s2", "approval_rate_s2", "grade_change", "total_approved"])

    return cols

