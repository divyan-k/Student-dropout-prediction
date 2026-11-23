"""Direct pipeline and data transformation tests."""

import pandas as pd
import numpy as np
import joblib

from src.config import MODEL_DIR
from src.features import load_data, engineer_features, get_feature_columns
from src.config import map_occupation


def test_occupation_mapping_coverage():
    """Verify that all unique parent occupation codes map cleanly to ISCO-08 groups."""
    df = load_data()
    for col in ["Mother's occupation", "Father's occupation"]:
        codes = df[col].dropna().unique()
        for c in codes:
            group = map_occupation(int(c))
            assert group in ("white_collar", "blue_collar", "other"), f"Code {c} mapped to invalid group {group}"


def test_pipeline_unseen_category_no_crash():
    """Verify handle_unknown='ignore' handles completely unseen category codes gracefully."""
    stage1_file = MODEL_DIR / "stage1.joblib"
    if not stage1_file.exists():
        return

    pipeline = joblib.load(stage1_file)
    cols = get_feature_columns(1)

    # Input with unseen codes (e.g. Course 99999, Nacionality 8888)
    dummy_input = pd.DataFrame([{
        col: 0 for col in cols
    }])
    dummy_input["Course"] = 99999
    dummy_input["Nacionality"] = 8888
    dummy_input["mother_occupation_group"] = "other"
    dummy_input["father_occupation_group"] = "other"

    # Should not crash
    probas = pipeline.predict_proba(dummy_input)
    assert probas.shape == (1, 3)
    assert np.isclose(np.sum(probas), 1.0)


def test_pipeline_deterministic():
    """Verify pipeline inference is fully deterministic on identical input."""
    stage3_file = MODEL_DIR / "stage3.joblib"
    if not stage3_file.exists():
        return

    pipeline = joblib.load(stage3_file)
    cols = get_feature_columns(3)

    sample = pd.DataFrame([{col: 1 for col in cols}])
    sample["mother_occupation_group"] = "white_collar"
    sample["father_occupation_group"] = "blue_collar"

    p1 = pipeline.predict_proba(sample)
    p2 = pipeline.predict_proba(sample)
    np.testing.assert_array_almost_equal(p1, p2)

