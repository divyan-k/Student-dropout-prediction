"""Preprocessing pipeline definitions per stage.

Builds sklearn ColumnTransformer and Pipeline objects:
- OneHotEncoder(handle_unknown="ignore", sparse_output=False) for categoricals
- StandardScaler for numeric continuous variables
- Passthrough for binary indicator variables
- Stratified train/test splitter
"""

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_COLS,
    BINARY_COLS,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features import get_feature_columns, engineer_features


def get_column_types(stage: int) -> Tuple[List[str], List[str], List[str]]:
    """Determine categorical, numeric, and binary column groups for a stage."""
    all_cols = get_feature_columns(stage)

    # Categoricals in this stage (raw minus replaced occupations + engineered groups)
    cats = [c for c in CATEGORICAL_COLS if c in all_cols]
    if "mother_occupation_group" in all_cols:
        cats.append("mother_occupation_group")
    if "father_occupation_group" in all_cols:
        cats.append("father_occupation_group")

    # Binary columns in this stage
    bins = [c for c in BINARY_COLS if c in all_cols]
    if "enrolled_zero_s1" in all_cols:
        bins.append("enrolled_zero_s1")
    if "enrolled_zero_s2" in all_cols:
        bins.append("enrolled_zero_s2")

    # Numeric columns: all remaining columns
    nums = [c for c in all_cols if c not in cats and c not in bins]

    return cats, nums, bins


def build_preprocessor(stage: int) -> ColumnTransformer:
    """Build ColumnTransformer for a given stage."""
    cats, nums, bins = get_column_types(stage)

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", StandardScaler(), nums),
        ("bin", "passthrough", bins),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(stage: int, model) -> Pipeline:
    """Assemble complete sklearn Pipeline (preprocessor + classifier)."""
    preprocessor = build_preprocessor(stage)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def split_data(
    df: pd.DataFrame,
    stage: int,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Engineer features for stage and perform leak-free stratified train-test split."""
    df_eng = engineer_features(df, stage=stage)
    feature_cols = get_feature_columns(stage)

    X = df_eng[feature_cols].copy()
    y = df_eng[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test

