"""Flask web application and REST API endpoints.

Implements all 9 endpoints defined in PROJECT_SPEC Section 11:
1. GET  /                          -> Web UI form
2. GET  /about                     -> Project & methodology info page
3. GET  /api/health                -> Health check & loaded model IDs
4. GET  /api/model-info            -> Model versions, metrics, class labels
5. GET  /api/features?stage=N      -> Features per stage with types & constraints
6. GET  /api/options               -> Readable label mappings for coded fields
7. POST /api/predict               -> Single student prediction & explainability
8. POST /api/predict/batch         -> Batch student prediction (up to 500 rows)
9. GET  /api/explain/global/<stage>-> Global feature importance
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from src.config import (
    MODEL_DIR,
    STAGE_FEATURES,
    STAGE1_PERFORMANCE_FIELDS,
    STAGE2_PERFORMANCE_FIELDS,
    STAGE3_PERFORMANCE_FIELDS,
    RANGES,
    BINARY_COLS,
    CATEGORICAL_COLS,
    TARGET_NAMES,
)
from src.labels import ALL_OPTIONS
from src.features import engineer_features, get_feature_columns
from src.risk import classify_risk, suggest_actions
from src.explain import get_top_factors, get_feature_importance
from app.schemas import (
    detect_stage,
    handle_forced_stage,
    fill_defaults,
    validate_student,
)

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.json = CustomJSONProvider(app)
CORS(app)

# Global caches for trained models and metadata
MODELS: Dict[int, Any] = {}
METADATA: Dict[str, Any] = {}


def load_application_models():
    """Load joblib pipelines and metadata.json with startup consistency check."""
    global MODELS, METADATA
    metadata_path = MODEL_DIR / "metadata.json"

    if not metadata_path.exists():
        logger.warning(f"metadata.json not found at {metadata_path}. Models may need training.")
        return

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read metadata.json: {e}")
        return

    for stage in (1, 2, 3):
        model_file = MODEL_DIR / f"stage{stage}.joblib"
        if model_file.exists():
            try:
                pipeline = joblib.load(model_file)
                MODELS[stage] = pipeline
                logger.info(f"Loaded Stage {stage} pipeline from {model_file}")
            except Exception as e:
                logger.error(f"Failed to load {model_file}: {e}")
        else:
            logger.warning(f"{model_file} not found.")

    logger.info(f"Startup check complete. Loaded stages: {list(MODELS.keys())}")


# Load models on server module import
load_application_models()


# ==============================================================================
# UI Pages
# ==============================================================================

@app.route("/", methods=["GET"])
def index_page():
    """Render main interactive prediction form."""
    return render_template("index.html")


@app.route("/about", methods=["GET"])
def about_page():
    """Render project documentation and model metrics page."""
    return render_template("about.html", metadata=METADATA)


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def api_health():
    """Service health check."""
    return jsonify({
        "status": "ok",
        "models_loaded": sorted(list(MODELS.keys())),
        "version": METADATA.get("version", "1.0.0"),
    }), 200


@app.route("/api/model-info", methods=["GET"])
@app.route("/api/info", methods=["GET"])
def api_model_info():
    """Model versions, metrics, and class definitions."""
    return jsonify({
        "version": METADATA.get("version", "1.0.0"),
        "trained_on": METADATA.get("trained_on"),
        "sklearn_version": METADATA.get("sklearn_version"),
        "shap_version": METADATA.get("shap_version"),
        "classes": METADATA.get("classes", TARGET_NAMES),
        "stages": METADATA.get("stages", {}),
    }), 200


@app.route("/api/options", methods=["GET"])
def api_options():
    """Dropdown option choices with human-readable labels."""
    formatted = {}
    for field, code_dict in ALL_OPTIONS.items():
        formatted[field] = [{"code": code, "label": label} for code, label in code_dict.items()]
    return jsonify(formatted), 200


@app.route("/api/features", methods=["GET"])
def api_features():
    """Feature list per stage with types and constraints."""
    stage_param = request.args.get("stage")
    stage = 1
    if stage_param:
        try:
            stage = int(stage_param)
            if stage not in (1, 2, 3):
                return jsonify({"error": "invalid_stage", "message": "Stage must be 1, 2, or 3."}), 400
        except ValueError:
            return jsonify({"error": "invalid_stage", "message": "Stage query param must be an integer."}), 400

    raw_features = STAGE_FEATURES[stage]
    feature_meta = []

    for f in raw_features:
        meta: Dict[str, Any] = {"name": f}
        if f in BINARY_COLS:
            meta["type"] = "binary"
            meta["allowed"] = [0, 1]
            meta["required"] = False
        elif f in CATEGORICAL_COLS:
            meta["type"] = "category"
            meta["allowed_source"] = "/api/options"
            meta["required"] = (f == "Course" or f == "Application mode")
        elif f in RANGES:
            meta["type"] = "numeric"
            meta["min"] = RANGES[f][0]
            meta["max"] = RANGES[f][1]
            meta["required"] = False
        elif "grade" in f:
            meta["type"] = "numeric"
            meta["min"] = 0.0
            meta["max"] = 20.0
            meta["required"] = True
        elif "Curricular units" in f:
            meta["type"] = "integer"
            meta["min"] = 0
            meta["required"] = True
        else:
            meta["type"] = "numeric"
            meta["required"] = False

        # Required performance fields per stage
        if stage == 2 and f in STAGE2_PERFORMANCE_FIELDS:
            meta["required"] = True
        if stage == 3 and (f in STAGE2_PERFORMANCE_FIELDS or f in STAGE3_PERFORMANCE_FIELDS):
            meta["required"] = True

        feature_meta.append(meta)

    return jsonify({"stage": stage, "features": feature_meta}), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Single student prediction endpoint."""
    if not request.is_json:
        return jsonify({"error": "bad_request", "message": "Request body must be valid JSON."}), 400

    data = request.get_json()
    if not data or "student" not in data:
        return jsonify({"error": "bad_request", "message": "Missing 'student' object in request body."}), 400

    student_data = data["student"]
    if not isinstance(student_data, dict):
        return jsonify({"error": "bad_request", "message": "'student' must be a JSON object."}), 400

    # 1. Determine Stage
    forced_stage = data.get("stage")
    ignored_fields = []
    if forced_stage is not None:
        try:
            stage_to_use = int(forced_stage)
            if stage_to_use not in (1, 2, 3):
                return jsonify({"error": "invalid_stage", "message": "Forced stage must be 1, 2, or 3."}), 400
            student_data, ignored_fields = handle_forced_stage(student_data, stage_to_use)
        except (ValueError, TypeError):
            return jsonify({"error": "invalid_stage", "message": "Forced stage must be an integer (1, 2, 3)."}), 400
    else:
        stage_to_use = detect_stage(student_data)

    if stage_to_use not in MODELS:
        return jsonify({
            "error": "service_unavailable",
            "message": f"Stage {stage_to_use} model is not loaded. Run 'python -m src.train' first.",
        }), 503

    # 2. Defaults for optional admission fields
    defaults_dict = METADATA.get("defaults", {})
    filled_data, defaulted_fields = fill_defaults(student_data, stage_to_use, defaults_dict)

    # 3. Validation
    cleaned_data, validation_errors = validate_student(filled_data, stage_to_use)
    if validation_errors:
        return jsonify({
            "error": "validation_error",
            "message": "One or more fields failed validation checks.",
            "details": validation_errors,
        }), 422

    pipeline = MODELS[stage_to_use]

    # 4. Feature engineering and formatting
    input_df = pd.DataFrame([cleaned_data])
    input_eng = engineer_features(input_df, stage=stage_to_use)
    expected_cols = get_feature_columns(stage_to_use)

    # Ensure all expected columns are present with appropriate default values
    for col in expected_cols:
        if col not in input_eng.columns:
            if "group" in col:
                input_eng[col] = "other"
            else:
                input_eng[col] = 0

    X_in = input_eng[expected_cols]

    # 5. Model Inference
    try:
        probas = pipeline.predict_proba(X_in)[0]
        pred_idx = int(np.argmax(probas))
        predicted_class = TARGET_NAMES[pred_idx]
        dropout_prob = float(probas[0])
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({"error": "inference_error", "message": f"Model inference error: {str(e)}"}), 500

    # 6. Risk level & Suggested Actions
    risk_level = classify_risk(dropout_prob)
    actions = suggest_actions(risk_level, cleaned_data)

    # 7. Explainability: Top Factors
    top_factors = get_top_factors(pipeline, input_eng, expected_cols, top_n=3)

    response = {
        "stage_used": stage_to_use,
        "prediction": predicted_class,
        "probabilities": {
            TARGET_NAMES[0]: round(float(probas[0]), 4),
            TARGET_NAMES[1]: round(float(probas[1]), 4),
            TARGET_NAMES[2]: round(float(probas[2]), 4),
        },
        "dropout_probability": round(dropout_prob, 4),
        "risk_level": risk_level,
        "suggested_actions": actions,
        "top_factors": top_factors,
        "defaulted_fields": defaulted_fields,
        "model_version": METADATA.get("version", "1.0.0"),
    }

    if ignored_fields:
        response["ignored_fields"] = ignored_fields
        response["note"] = f"Fields {ignored_fields} were ignored because stage {stage_to_use} was explicitly requested."

    return jsonify(response), 200


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """Batch prediction endpoint for up to 500 students with a uniform stage."""
    if not request.is_json:
        return jsonify({"error": "bad_request", "message": "Request body must be valid JSON."}), 400

    data = request.get_json()
    if not data or "students" not in data:
        return jsonify({"error": "bad_request", "message": "Missing 'students' list in request body."}), 400

    students_list = data["students"]
    if not isinstance(students_list, list):
        return jsonify({"error": "bad_request", "message": "'students' must be a list."}), 400

    if len(students_list) > 500:
        return jsonify({
            "error": "payload_too_large",
            "message": f"Batch limit exceeded. Max 500 students allowed, received {len(students_list)}.",
        }), 413

    stage_req = data.get("stage", 1)
    try:
        stage = int(stage_req)
        if stage not in (1, 2, 3):
            return jsonify({"error": "invalid_stage", "message": "Batch stage must be 1, 2, or 3."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "invalid_stage", "message": "Stage must be an integer."}), 400

    if stage not in MODELS:
        return jsonify({
            "error": "service_unavailable",
            "message": f"Stage {stage} model is not loaded.",
        }), 503

    pipeline = MODELS[stage]
    defaults_dict = METADATA.get("defaults", {})
    expected_cols = get_feature_columns(stage)

    results = []
    errors = []

    for idx, student in enumerate(students_list):
        if not isinstance(student, dict):
            errors.append({"index": idx, "message": "Row item must be a JSON object."})
            continue

        trimmed_student, _ = handle_forced_stage(student, stage)
        filled, _ = fill_defaults(trimmed_student, stage, defaults_dict)
        cleaned, val_errors = validate_student(filled, stage)

        if val_errors:
            errors.append({"index": idx, "message": "Validation failed", "details": val_errors})
            continue

        try:
            row_df = pd.DataFrame([cleaned])
            row_eng = engineer_features(row_df, stage=stage)
            for c in expected_cols:
                if c not in row_eng.columns:
                    if "group" in c:
                        row_eng[c] = "other"
                    else:
                        row_eng[c] = 0
            probas = pipeline.predict_proba(row_eng[expected_cols])[0]
            pred_idx = int(np.argmax(probas))
            dropout_prob = float(probas[0])
            risk_level = classify_risk(dropout_prob)

            results.append({
                "index": idx,
                "prediction": TARGET_NAMES[pred_idx],
                "dropout_probability": round(dropout_prob, 4),
                "risk_level": risk_level,
            })
        except Exception as e:
            errors.append({"index": idx, "message": f"Inference failed: {str(e)}"})

    return jsonify({
        "stage_used": stage,
        "count": len(results),
        "results": results,
        "errors": errors,
    }), 200


@app.route("/api/explain/global/<int:stage>", methods=["GET"])
def api_explain_global(stage: int):
    """Global feature importance for a given stage."""
    if stage not in (1, 2, 3):
        return jsonify({"error": "not_found", "message": "Stage must be 1, 2, or 3."}), 404

    if stage not in MODELS:
        return jsonify({"error": "service_unavailable", "message": f"Stage {stage} model not loaded."}), 503

    pipeline = MODELS[stage]
    expected_cols = get_feature_columns(stage)
    importances = get_feature_importance(pipeline, expected_cols)

    return jsonify({
        "stage": stage,
        "method": "aggregated_model_importance",
        "top_features": importances[:15],
    }), 200


@app.route("/api/explain/fairness", methods=["GET"])
def api_explain_fairness():
    """Return subgroup demographic fairness audit."""
    if not METADATA:
        return jsonify({"error": "service_unavailable", "message": "Metadata not loaded."}), 503

    stage_arg = request.args.get("stage")
    if stage_arg == "2":
        audit_data = METADATA.get("fairness_audit_stage2", [])
        stage_num = 2
    else:
        audit_data = METADATA.get("fairness_audit_stage3", METADATA.get("fairness_audit", []))
        stage_num = 3

    return jsonify({
        "status": "success",
        "stage": stage_num,
        "description": f"Subgroup fairness audit on Stage {stage_num} held-out test partition across sensitive demographic and socio-economic attributes.",
        "fairness_audit": audit_data,
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

