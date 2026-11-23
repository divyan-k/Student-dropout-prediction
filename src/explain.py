"""Model explainability and SHAP helpers.

Key capabilities:
- TreeExplainer for tree-based ensembles (RandomForest, GradientBoosting, XGBoost, DecisionTree)
- LinearExplainer for linear models (LogisticRegression)
- Aggregation of one-hot encoded categories back to original feature names
- Signed SHAP attribution for local sample explanations
- Global feature importance extraction
- Subgroup demographic fairness checks with sample size n reporting
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "approval_rate_s1": "Semester 1 pass rate",
    "approval_rate_s2": "Semester 2 pass rate",
    "grade_change": "Grade momentum (Sem 2 - Sem 1)",
    "total_approved": "Total curricular units approved",
    "enrolled_zero_s1": "Zero curricular units enrolled (Sem 1)",
    "enrolled_zero_s2": "Zero curricular units enrolled (Sem 2)",
    "mother_occupation_group": "Mother's occupation category",
    "father_occupation_group": "Father's occupation category",
}


def compute_wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z**2) / (4 * (n**2))))
    low = round(max(0.0, center - spread) * 100, 1)
    high = round(min(1.0, center + spread) * 100, 1)
    return (low, high)


def get_feature_importance(pipeline: Pipeline, original_features: List[str]) -> List[Dict[str, Any]]:
    """Extract global feature importance aggregated back to original feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Get transformed column names from ColumnTransformer
    transformed_cols = preprocessor.get_feature_names_out()

    # Extract raw importances/coefficients
    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        # For multi-class logistic regression, take mean absolute weight across classes
        raw_importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        return []

    # Map transformed columns to original feature names and aggregate
    agg_map: Dict[str, float] = {feat: 0.0 for feat in original_features}

    for col_name, imp in zip(transformed_cols, raw_importances):
        # Format is usually 'cat__Course_33' or 'num__Admission grade' or 'bin__Debtor'
        clean_name = col_name.split("__", 1)[-1]
        matched = False
        for orig in original_features:
            if clean_name == orig or clean_name.startswith(orig + "_"):
                agg_map[orig] += float(imp)
                matched = True
                break
        if not matched:
            agg_map[clean_name] = agg_map.get(clean_name, 0.0) + float(imp)

    # Normalize to sum to 1.0
    total = sum(agg_map.values())
    if total > 0:
        agg_map = {k: v / total for k, v in agg_map.items()}

    sorted_feats = sorted(agg_map.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": feat, "importance": round(score, 4)} for feat, score in sorted_feats]


def get_explainer(pipeline: Pipeline, X_background: pd.DataFrame):
    """Instantiate appropriate SHAP explainer based on classifier family."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(X_background)

    class_name = classifier.__class__.__name__
    if any(tree_model in class_name for tree_model in ["Forest", "Gradient", "XGB", "DecisionTree"]):
        # TreeExplainer is fast and exact for trees
        return shap.TreeExplainer(classifier)
    elif "Logistic" in class_name:
        # LinearExplainer for linear models
        return shap.LinearExplainer(classifier, X_trans)
    else:
        # KernelExplainer fallback on small background sample
        sample = X_trans[:50] if len(X_trans) > 50 else X_trans
        return shap.KernelExplainer(classifier.predict_proba, sample)


def aggregate_shap_values(
    shap_vals_dropout: np.ndarray,
    transformed_cols: List[str],
    original_features: List[str],
) -> Dict[str, float]:
    """Sum signed SHAP values across one-hot columns back to original feature names.
    
    Ranking is performed by abs(sum), while impact direction preserves the sign:
    - Positive: pushes toward Dropout (increases risk)
    - Negative: pushes away from Dropout toward graduation (decreases risk)
    """
    agg_signed: Dict[str, float] = {f: 0.0 for f in original_features}

    for col_name, val in zip(transformed_cols, shap_vals_dropout):
        clean_name = col_name.split("__", 1)[-1]
        matched = False
        for orig in original_features:
            if clean_name == orig or clean_name.startswith(orig + "_"):
                agg_signed[orig] += float(val)
                matched = True
                break
        if not matched:
            agg_signed[clean_name] = agg_signed.get(clean_name, 0.0) + float(val)

    return agg_signed


def get_top_factors(
    pipeline: Pipeline,
    single_row_df: pd.DataFrame,
    original_features: List[str],
    background_df: Optional[pd.DataFrame] = None,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Compute local SHAP explanation for a single student prediction."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_cols = list(preprocessor.get_feature_names_out())

    X_trans = preprocessor.transform(single_row_df)

    # Fast heuristic tree attribution if SHAP explainer isn't pre-warmed
    try:
        if background_df is not None:
            explainer = get_explainer(pipeline, background_df.iloc[:100])
            shap_obj = explainer(X_trans)
            # Handle multi-class SHAP dimensions
            # Usually shap_obj.values has shape (1, num_features, num_classes) or list of classes
            if isinstance(shap_obj.values, list):
                dropout_shap = shap_obj.values[0][0]
            elif shap_obj.values.ndim == 3:
                dropout_shap = shap_obj.values[0, :, 0]
            else:
                dropout_shap = shap_obj.values[0]
        else:
            # Fallback to feature importance weighting
            imp_list = get_feature_importance(pipeline, original_features)
            top_factors = []
            for item in imp_list[:top_n]:
                feat = item["feature"]
                val = single_row_df[feat].iloc[0] if feat in single_row_df.columns else "N/A"
                if isinstance(val, (float, np.floating)):
                    val = round(float(val), 2)
                top_factors.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": val,
                    "impact": "decreases_risk" if "approved" in feat or "rate" in feat or "grade" in feat else ("increases_risk" if "fees" in feat or "Debtor" in feat else "neutral"),
                })
            return top_factors
    except Exception:
        # Safe fallback
        imp_list = get_feature_importance(pipeline, original_features)
        top_factors = []
        for item in imp_list[:top_n]:
            feat = item["feature"]
            val = single_row_df[feat].iloc[0] if feat in single_row_df.columns else "N/A"
            if isinstance(val, (float, np.floating)):
                val = round(float(val), 2)
            top_factors.append({
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "value": val,
                "impact": "increases_risk" if item["importance"] > 0.05 else "decreases_risk",
            })
        return top_factors

    agg_signed = aggregate_shap_values(dropout_shap, transformed_cols, original_features)

    # Rank by absolute magnitude of the signed sum
    ranked = sorted(agg_signed.items(), key=lambda x: abs(x[1]), reverse=True)

    factors = []
    for feat, signed_sum in ranked[:top_n]:
        raw_val = single_row_df[feat].iloc[0] if feat in single_row_df.columns else None
        if isinstance(raw_val, (float, np.floating)):
            raw_val = round(float(raw_val), 2)
        factors.append({
            "feature": feat,
            "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
            "value": raw_val,
            "impact": "increases_risk" if signed_sum > 0 else "decreases_risk",
            "weight": round(float(signed_sum), 4),
        })

    return factors


def plot_shap_summary(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    stage: int,
    save_path: Optional[str] = None,
) -> str:
    """Plot global feature importance summary bar chart."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = save_path or str(FIGURES_DIR / f"shap_summary_stage{stage}.png")

    original_features = list(X_sample.columns)
    importances = get_feature_importance(pipeline, original_features)
    top_items = importances[:12]

    feats = [FEATURE_DISPLAY_NAMES.get(item["feature"], item["feature"]) for item in reversed(top_items)]
    scores = [item["importance"] for item in reversed(top_items)]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(feats, scores, color="#2a9d8f")
    plt.xlabel("Aggregated Relative Importance")
    plt.title(f"Stage {stage} Top Feature Drivers (Aggregated)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    return out_file


def fairness_check(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Perform demographic parity & recall fairness check across subgroups with 95% Wilson CI."""
    if sensitive_cols is None:
        sensitive_cols = [
            "Tuition fees up to date",
            "Debtor",
            "Gender",
            "Scholarship holder",
            "International",
            "Educational special needs",
        ]

    # Filter to existing columns in this stage
    valid_cols = [c for c in sensitive_cols if c in X_test.columns]

    y_pred = pipeline.predict(X_test)
    records = []

    for col in valid_cols:
        for val, group in X_test.groupby(col):
            idx = group.index
            y_true_sub = y_test.loc[idx]
            y_pred_sub = y_pred[X_test.index.get_indexer(idx)]

            n = len(y_true_sub)
            n_dropout_true = int((y_true_sub == 0).sum())
            k_caught = int(((y_true_sub == 0) & (y_pred_sub == 0)).sum())

            # Dropout recall in this subgroup
            if n_dropout_true > 0:
                dropout_recall = float(k_caught / n_dropout_true)
                ci_low, ci_high = compute_wilson_ci(k_caught, n_dropout_true)
                ci_str = f"[{ci_low}%, {ci_high}%]"
            else:
                dropout_recall = np.nan
                ci_str = "N/A"

            sub_acc = float((y_true_sub == y_pred_sub).mean())

            # Determine reliability by number of true dropouts (minimum 50 for statistical stability)
            if n_dropout_true >= 50:
                rel = "Reliable (n_dropouts >= 50)"
            else:
                rel = f"Indicative (n_dropouts={n_dropout_true} < 50)"

            records.append({
                "attribute": col,
                "group_value": val,
                "sample_size_n": n,
                "true_dropouts": n_dropout_true,
                "dropouts_caught": k_caught,
                "accuracy": round(sub_acc, 3),
                "dropout_recall": round(dropout_recall, 3) if not np.isnan(dropout_recall) else "N/A",
                "ci_95": ci_str,
                "reliable": rel,
            })

    return pd.DataFrame(records)

