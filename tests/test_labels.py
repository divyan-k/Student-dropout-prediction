"""Coverage and integrity tests for hand-written code-to-label dictionaries."""

import pandas as pd
from src.labels import ALL_OPTIONS, VALID_CODES
from src.features import load_data


def test_labels_format():
    """Verify dictionary entries are valid int keys and non-empty string labels."""
    for field_name, mapping in ALL_OPTIONS.items():
        assert len(mapping) > 0, f"Mapping for {field_name} is empty."
        for code, label in mapping.items():
            assert isinstance(code, int), f"Code {code} in {field_name} is not int."
            assert isinstance(label, str) and len(label.strip()) > 0, (
                f"Label for code {code} in {field_name} is invalid."
            )


def test_labels_data_coverage():
    """Verify every unique code present in the training dataset has an entry in ALL_OPTIONS."""
    df = load_data()

    for field in [
        "Course",
        "Application mode",
        "Marital Status",
        "Gender",
        "Daytime/evening attendance",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "International",
        "Nacionality",
        "Previous qualification",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ]:
        if field in df.columns:
            unique_codes = df[field].dropna().unique()
            for code in unique_codes:
                assert int(code) in VALID_CODES[field], (
                    f"Code {code} for '{field}' in dataset is missing from labels.py!"
                )

