"""Integration tests for Flask REST API endpoints."""

import pytest
import numpy as np
from app.main import app, load_application_models


@pytest.fixture
def client():
    """Create test client for Flask application."""
    load_application_models()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """GET /api/health should return status ok and list of loaded models."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_model_info(client):
    """GET /api/model-info should return model version, classes, and stages."""
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert "version" in data
    assert "classes" in data
    assert "stages" in data


def test_options_endpoint(client):
    """GET /api/options should return label lists for all categorical attributes."""
    res = client.get("/api/options")
    assert res.status_code == 200
    data = res.get_json()
    assert "Course" in data
    assert len(data["Course"]) == 17
    assert "Application mode" in data


def test_valid_stage1_prediction(client):
    """POST /api/predict with admission data should return Stage 1 prediction."""
    payload = {
        "student": {
            "Course": 9254,
            "Application mode": 1,
            "Admission grade": 130.0,
            "Previous qualification (grade)": 135.0,
            "Age at enrollment": 20,
            "Gender": 1,
            "Tuition fees up to date": 1,
            "Debtor": 0,
            "Scholarship holder": 0,
            "Daytime/evening attendance": 1,
        }
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["stage_used"] == 1
    assert data["prediction"] in ("Dropout", "Enrolled", "Graduate")
    assert "risk_level" in data
    assert "suggested_actions" in data
    assert "top_factors" in data

    probs = data["probabilities"]
    prob_sum = probs["Dropout"] + probs["Enrolled"] + probs["Graduate"]
    assert np.isclose(prob_sum, 1.0, atol=1e-2)


def test_valid_stage3_prediction(client):
    """POST /api/predict with full semester data should trigger Stage 3."""
    payload = {
        "student": {
            "Course": 9500,
            "Application mode": 1,
            "Admission grade": 140.0,
            "Previous qualification (grade)": 140.0,
            "Age at enrollment": 19,
            "Gender": 0,
            "Tuition fees up to date": 1,
            "Debtor": 0,
            "Scholarship holder": 1,
            "Daytime/evening attendance": 1,
            "Curricular units 1st sem (enrolled)": 6,
            "Curricular units 1st sem (approved)": 6,
            "Curricular units 1st sem (evaluations)": 6,
            "Curricular units 1st sem (grade)": 15.0,
            "Curricular units 2nd sem (enrolled)": 6,
            "Curricular units 2nd sem (approved)": 6,
            "Curricular units 2nd sem (evaluations)": 6,
            "Curricular units 2nd sem (grade)": 15.5,
        }
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["stage_used"] == 3
    assert data["risk_level"] == "Low"


def test_missing_student_key(client):
    """Missing 'student' wrapper key should return 400 Bad Request."""
    res = client.post("/api/predict", json={"invalid_key": {}})
    assert res.status_code == 400


def test_invalid_age_validation(client):
    """Age 200 should fail schema validation and return 422."""
    payload = {
        "student": {
            "Course": 9254,
            "Application mode": 1,
            "Admission grade": 120.0,
            "Age at enrollment": 200,
        }
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
    data = res.get_json()
    assert data["error"] == "validation_error"


def test_approved_exceeding_enrolled(client):
    """Approved units exceeding enrolled units should return 422."""
    payload = {
        "student": {
            "Course": 9254,
            "Application mode": 1,
            "Admission grade": 120.0,
            "Age at enrollment": 20,
            "Curricular units 1st sem (enrolled)": 5,
            "Curricular units 1st sem (approved)": 10,
            "Curricular units 1st sem (evaluations)": 10,
            "Curricular units 1st sem (grade)": 12.0,
        }
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422


def test_forced_stage_ignores_extras(client):
    """Forced Stage 1 should ignore provided Semester 2 performance fields."""
    payload = {
        "stage": 1,
        "student": {
            "Course": 9254,
            "Application mode": 1,
            "Admission grade": 120.0,
            "Age at enrollment": 20,
            "Curricular units 2nd sem (enrolled)": 6,
            "Curricular units 2nd sem (approved)": 6,
        }
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["stage_used"] == 1
    assert "ignored_fields" in data


def test_batch_prediction(client):
    """POST /api/predict/batch with multiple records."""
    payload = {
        "stage": 1,
        "students": [
            {
                "Course": 9254,
                "Application mode": 1,
                "Admission grade": 120.0,
                "Age at enrollment": 20,
            },
            {
                "Course": 9500,
                "Application mode": 1,
                "Admission grade": 140.0,
                "Age at enrollment": 21,
            },
        ]
    }
    res = client.post("/api/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_batch_exceeding_limit(client):
    """Batch over 500 rows should return 413 Payload Too Large."""
    payload = {
        "stage": 1,
        "students": [{"Course": 9254} for _ in range(501)],
    }
    res = client.post("/api/predict/batch", json=payload)
    assert res.status_code == 413


def test_explain_fairness_endpoint(client):
    """GET /api/explain/fairness returns subgroup demographic parity metrics."""
    res = client.get("/api/explain/fairness")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "fairness_audit" in data
    assert len(data["fairness_audit"]) > 0
    # Check sample size n and reliability fields exist
    first_row = data["fairness_audit"][0]
    assert "sample_size_n" in first_row
    assert "reliable" in first_row


def test_options_completeness_and_ui_sync(client):
    """Verify that all 17 option categories are present in /api/options and index.html."""
    import re
    from pathlib import Path
    from src.labels import ALL_OPTIONS

    res = client.get("/api/options")
    assert res.status_code == 200
    options = res.get_json()

    # All 17 option groups must be populated
    assert len(options) == 17
    for field_name, items in options.items():
        assert len(items) > 0, f"Options for '{field_name}' must not be empty"
        for opt in items:
            assert "code" in opt and "label" in opt

    # Check that index.html contains all 17 select IDs
    index_html_path = Path(__file__).resolve().parent.parent / "app" / "templates" / "index.html"
    html_text = index_html_path.read_text(encoding="utf-8")
    select_ids = set(re.findall(r'<select[^>]+id=[\'"]([^\'"]+)[\'"]', html_text))

    app_js_path = Path(__file__).resolve().parent.parent / "app" / "static" / "app.js"
    js_text = app_js_path.read_text(encoding="utf-8")

    for field in ALL_OPTIONS.keys():
        assert field in options
        # Match from FIELD_ID_MAP in app.js
        pattern = rf'"{re.escape(field)}":\s*"([^"]+)"'
        match = re.search(pattern, js_text)
        assert match is not None, f"Mapping missing in JS FIELD_ID_MAP for '{field}'"
        mapped_id = match.group(1)
        assert mapped_id in select_ids, f"Element ID '{mapped_id}' not found in index.html"


