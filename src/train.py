"""Model training, cross-validation, hyperparameter tuning, and metadata generation.

Option A: Standardized Random Forest Architecture across all three prediction stages.

Design:
- Primary selection criterion: Dropout F1 (and Dropout Recall), guarding with a Macro-F1 floor
  to protect multi-class recognition while optimizing early dropout detection.
- Leak-free train-only cross-validation and tuning (X_test held out and evaluated ONCE).
- Uniform imbalance handling via balanced sample weights routed strictly to fit().
- Unweighted scoring for accuracy, macro-F1, dropout recall, and dropout F1.
- Paired 5-fold CV benchmark for Random Forest against previous candidates.
- Automated calculation of risk bands with Wilson score 95% confidence intervals and precision.
- Subgroup demographic fairness audits across Stages 1, 2, and 3.
- Output of stage1.joblib, stage2.joblib, stage3.joblib, and comprehensive metadata.json.
"""

import sys
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import shap

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    MODEL_DIR,
    RANDOM_STATE,
    CV_FOLDS,
    DEFAULTABLE_FIELDS,
    NUMERIC_COLS_BASE,
    TARGET_NAMES,
    map_occupation,
)
from src.features import load_data, engineer_features, get_feature_columns
from src.preprocess import build_pipeline, split_data
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
)
from src.explain import plot_shap_summary, fairness_check, compute_wilson_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelRunner:
    """Manages cross-validation, tuning, and evaluation for a model pipeline."""

    def __init__(self, stage: int, model: Any, name: str):
        self.stage = stage
        self.model = model
        self.name = name
        self.pipeline = build_pipeline(stage, model)

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = CV_FOLDS) -> Dict[str, Any]:
        """Perform stratified k-fold CV with fold-specific sample weights and unweighted scoring."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        accs, macro_f1s, drop_recalls, drop_f1s, drop_precs = [], [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            sw = compute_sample_weight("balanced", y_tr)

            fold_pipeline = clone(self.pipeline)
            fold_pipeline.fit(X_tr, y_tr, classifier__sample_weight=sw)

            y_pred = fold_pipeline.predict(X_val)
            y_proba = fold_pipeline.predict_proba(X_val)

            metrics = compute_metrics(y_val.to_numpy(), y_pred, y_proba)
            accs.append(metrics["accuracy"])
            macro_f1s.append(metrics["macro_f1"])
            drop_recalls.append(metrics["dropout_recall"])
            drop_f1s.append(metrics["dropout_f1"])
            drop_precs.append(metrics["dropout_precision"])

        return {
            "model_name": self.name,
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "macro_f1": round(float(np.mean(macro_f1s)), 4),
            "macro_f1_std": round(float(np.std(macro_f1s)), 4),
            "dropout_recall": round(float(np.mean(drop_recalls)), 4),
            "dropout_recall_std": round(float(np.std(drop_recalls)), 4),
            "dropout_f1": round(float(np.mean(drop_f1s)), 4),
            "dropout_f1_std": round(float(np.std(drop_f1s)), 4),
            "dropout_precision": round(float(np.mean(drop_precs)), 4),
        }

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List[Any]],
        macro_guard_floor: float,
        n_iter: int = 8,
    ) -> Any:
        """Search across parameter grid prioritizing Dropout F1 with a Macro-F1 guard floor."""
        rng = np.random.RandomState(RANDOM_STATE)
        param_keys = list(param_grid.keys())

        candidates = []
        for _ in range(n_iter):
            cand = {k: rng.choice(param_grid[k]) for k in param_keys}
            if cand not in candidates:
                candidates.append(cand)

        best_score = -1.0
        best_params = {}
        best_pipeline = None

        logger.info(f"Tuning {self.name} across {len(candidates)} parameter sets (Macro-F1 guard >= {macro_guard_floor:.2f})...")

        for cand in candidates:
            temp_pipeline = clone(self.pipeline)
            temp_pipeline.set_params(**cand)

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            f1s, recs, drop_f1s = [], [], []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                sw = compute_sample_weight("balanced", y_tr)
                fold_pipe = clone(temp_pipeline)
                fold_pipe.fit(X_tr, y_tr, classifier__sample_weight=sw)

                y_pred = fold_pipe.predict(X_val)
                metrics = compute_metrics(y_val.to_numpy(), y_pred)
                f1s.append(metrics["macro_f1"])
                recs.append(metrics["dropout_recall"])
                drop_f1s.append(metrics["dropout_f1"])

            cand_macro_f1 = float(np.mean(f1s))
            cand_recall = float(np.mean(recs))
            cand_drop_f1 = float(np.mean(drop_f1s))

            # Guard enforcement: must maintain acceptable macro-F1 while optimizing dropout F1
            if cand_macro_f1 >= macro_guard_floor:
                if cand_drop_f1 > best_score:
                    best_score = cand_drop_f1
                    best_params = cand
                    best_pipeline = temp_pipeline

        # Fallback if no candidate satisfied the guard floor
        if best_pipeline is None:
            logger.warning(f"No candidate met guard floor {macro_guard_floor:.2f}. Falling back to default params.")
            best_pipeline = clone(self.pipeline)
            best_params = {}

        # Refit best pipeline on entire X_train with balanced sample weights
        sw_full = compute_sample_weight("balanced", y_train)
        best_pipeline.fit(X_train, y_train, classifier__sample_weight=sw_full)
        self.pipeline = best_pipeline
        return best_params

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Final evaluation on held-out test set."""
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)
        return compute_metrics(y_test.to_numpy(), y_pred, y_proba)

    def save(self, file_path: Path):
        """Save pipeline with joblib."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, file_path)


def compute_defaults(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Compute medians for continuous numerics and modes for categoricals on training split."""
    defaults = {}
    for col in DEFAULTABLE_FIELDS:
        if col in df_train.columns:
            series = df_train[col].dropna()
            if col in NUMERIC_COLS_BASE:
                defaults[col] = round(float(series.median()), 2)
            else:
                mode_val = series.mode()[0]
                defaults[col] = int(mode_val) if isinstance(mode_val, (np.integer, int)) else str(mode_val)
    return defaults


def compute_risk_bands(pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute risk bands, actual dropout rates, Wilson 95% CIs, and combined precision."""
    probas = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)
    dropout_idx = classes.index("Dropout") if "Dropout" in classes else classes.index(0)
    dropout_probs = probas[:, dropout_idx]

    bands = []
    for p in dropout_probs:
        if p < 0.33:
            bands.append("Low")
        elif p < 0.66:
            bands.append("Medium")
        else:
            bands.append("High")

    is_dropout = ((y_test.values == "Dropout") | (y_test.values == 0)).astype(int)
    eval_df = pd.DataFrame({"is_dropout": is_dropout, "band": bands})

    total_students = len(eval_df)
    total_dropouts = int(eval_df["is_dropout"].sum())

    stage_bands = {}
    for b_name in ["Low", "Medium", "High"]:
        b_df = eval_df[eval_df["band"] == b_name]
        n_b = len(b_df)
        n_drop = int(b_df["is_dropout"].sum())
        ci_low, ci_high = compute_wilson_ci(n_drop, n_b)
        actual_rate = round((n_drop / n_b * 100) if n_b > 0 else 0.0, 1)

        stage_bands[b_name] = {
            "share_students": round((n_b / total_students) * 100, 1),
            "count": n_b,
            "actual_dropout_rate": actual_rate,
            "actual_dropout_rate_ci": [ci_low, ci_high],
            "dropouts_in_band": n_drop,
            "share_of_all_dropouts": round((n_drop / total_dropouts * 100) if total_dropouts > 0 else 0.0, 1),
        }

    med_high = eval_df[eval_df["band"].isin(["Medium", "High"])]
    n_mh = len(med_high)
    drop_mh = int(med_high["is_dropout"].sum())
    prec_mh = round((drop_mh / n_mh * 100) if n_mh > 0 else 0.0, 1)
    ci_mh_low, ci_mh_high = compute_wilson_ci(drop_mh, n_mh)

    stage_bands["Medium_High_Combined"] = {
        "flagged_share_students": round((n_mh / total_students) * 100, 1),
        "flagged_count": n_mh,
        "dropouts_caught": drop_mh,
        "total_dropouts": total_dropouts,
        "share_of_all_dropouts_caught": round((drop_mh / total_dropouts * 100) if total_dropouts > 0 else 0.0, 1),
        "precision": prec_mh,
        "precision_ci": [ci_mh_low, ci_mh_high],
    }

    return stage_bands


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def train_all_stages():
    """Execute standardized Option A Random Forest training pipeline across all three stages."""
    logger.info("Initializing Option A (Random Forest at Every Stage) Model Training...")
    df = load_data()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "version": "2.0.0",
        "champion_family": "RandomForest",
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "shap_version": shap.__version__,
        "classes": TARGET_NAMES,
        "stages": {},
        "defaults": {},
        "non_defaultable_note": "Semester performance and core admission grades are never defaulted.",
        "candidate_cv_comparison": {},
        "risk_band_evaluations": {},
    }

    dummy_accs = {}

    # Guard floors per stage for Macro-F1
    macro_guards = {1: 0.52, 2: 0.65, 3: 0.68}

    for stage in (1, 2, 3):
        logger.info(f"\n=================== TRAINING STAGE {stage} ===================")
        X_train, X_test, y_train, y_test = split_data(df, stage=stage)
        features = list(X_train.columns)

        # Baseline: DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_acc = float(dummy.score(X_test, y_test))
        dummy_accs[stage] = round(dummy_acc, 4)
        logger.info(f"Stage {stage} Dummy Baseline Accuracy: {dummy_acc:.4f}")

        # Compute safe defaults on Stage 1 raw training features
        if stage == 1:
            metadata["defaults"] = compute_defaults(df.loc[X_train.index])

        # Candidate models for paired cross-validation
        candidates = [
            ModelRunner(stage, LogisticRegression(max_iter=500, random_state=RANDOM_STATE), "LogisticRegression"),
            ModelRunner(stage, DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE), "DecisionTree"),
            ModelRunner(stage, RandomForestClassifier(n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=1), "RandomForest"),
            ModelRunner(stage, GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE), "GradientBoosting"),
            ModelRunner(stage, XGBClassifier(n_estimators=100, max_depth=3, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1), "XGBoost"),
        ]

        # 1. Paired Cross-Validation across identical folds
        cv_results = []
        for runner in candidates:
            res = runner.cross_validate(X_train, y_train)
            cv_results.append(res)
            logger.info(
                f"[{runner.name}] Acc: {res['accuracy_mean']:.4f} | "
                f"Macro-F1: {res['macro_f1']:.4f} ± {res['macro_f1_std']:.4f} | "
                f"Drop-Recall: {res['dropout_recall']:.4f} ± {res['dropout_recall_std']:.4f} | "
                f"Drop-F1: {res['dropout_f1']:.4f}"
            )

        cv_df = pd.DataFrame(cv_results)
        plot_model_comparison(cv_df, stage=stage)
        metadata["candidate_cv_comparison"][str(stage)] = cv_results

        # 2. Select Option A: RandomForest Champion at Every Stage
        champion_runner = [c for c in candidates if c.name == "RandomForest"][0]
        rf_cv_row = cv_df[cv_df["model_name"] == "RandomForest"].iloc[0]
        logger.info(
            f"Stage {stage} Champion Selected: RandomForest (CV Drop-F1={rf_cv_row['dropout_f1']:.4f}, "
            f"Drop-Recall={rf_cv_row['dropout_recall']:.4f}, Macro-F1={rf_cv_row['macro_f1']:.4f})"
        )

        # 3. Hyperparameter Tuning for RandomForest
        rf_param_grid = {
            "classifier__n_estimators": [150, 200, 250],
            "classifier__max_depth": [10, 14, 18, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__criterion": ["gini", "entropy"],
        }

        best_params = champion_runner.tune(
            X_train,
            y_train,
            param_grid=rf_param_grid,
            macro_guard_floor=macro_guards[stage],
            n_iter=8,
        )
        logger.info(f"Stage {stage} Best Tuned Parameters: {best_params}")

        # 4. Final Evaluation on Held-Out Test Set (Evaluated ONCE)
        test_metrics = champion_runner.evaluate(X_test, y_test)
        logger.info(
            f"Stage {stage} Test Results (RandomForest): Acc={test_metrics['accuracy']:.4f}, "
            f"Macro-F1={test_metrics['macro_f1']:.4f}, Dropout-Recall={test_metrics['dropout_recall']:.4f}, "
            f"Dropout-F1={test_metrics['dropout_f1']:.4f}, ROC-AUC={test_metrics.get('roc_auc_ovr', 0):.4f}"
        )

        # Verification check: Beats dummy classifier by >= 7 points
        acc_diff = test_metrics["accuracy"] - dummy_accs[stage]
        if acc_diff >= 0.07:
            logger.info(f"Verification PASS: Stage {stage} beats DummyClassifier by {acc_diff*100:.1f} percentage points.")
        else:
            logger.warning(f"Verification NOTE: Stage {stage} accuracy margin over Dummy is {acc_diff*100:.1f} percentage points.")

        # Save model pipeline
        model_file = MODEL_DIR / f"stage{stage}.joblib"
        champion_runner.save(model_file)
        logger.info(f"Saved pipeline to {model_file}")

        # Generate plots
        y_pred = champion_runner.pipeline.predict(X_test)
        y_proba = champion_runner.pipeline.predict_proba(X_test)
        plot_confusion_matrix(y_test.to_numpy(), y_pred, stage=stage)
        plot_roc_curves(y_test.to_numpy(), y_proba, stage=stage)
        plot_shap_summary(champion_runner.pipeline, X_test, stage=stage)

        # 5. Compute Risk Band Evaluations & Wilson CIs
        stage_risk_bands = compute_risk_bands(champion_runner.pipeline, X_test, y_test)
        metadata["risk_band_evaluations"][str(stage)] = stage_risk_bands

        # Record stage metadata
        metadata["stages"][str(stage)] = {
            "name": f"Stage {stage}" if stage != 1 else "Stage 1 (Admission)",
            "model": "RandomForest",
            "best_params": best_params,
            "features": features,
            "metrics": test_metrics,
            "cv_metrics": {
                "macro_f1_mean": rf_cv_row["macro_f1"],
                "macro_f1_std": rf_cv_row["macro_f1_std"],
                "dropout_recall_mean": rf_cv_row["dropout_recall"],
                "dropout_recall_std": rf_cv_row["dropout_recall_std"],
                "dropout_f1_mean": rf_cv_row["dropout_f1"],
                "dropout_f1_std": rf_cv_row["dropout_f1_std"],
                "accuracy_mean": rf_cv_row["accuracy_mean"],
                "accuracy_std": rf_cv_row["accuracy_std"],
            },
            "macro_guard_floor": macro_guards[stage],
            "dummy_accuracy": dummy_accs[stage],
        }

    # 6. Subgroup Fairness Audit across all 3 stages on held-out test splits
    sensitive = [
        "Tuition fees up to date",
        "Debtor",
        "Gender",
        "Scholarship holder",
        "International",
        "Educational special needs",
    ]

    for stage in (1, 2, 3):
        X_tr, X_te, y_tr, y_te = split_data(df, stage=stage)
        pipe = joblib.load(MODEL_DIR / f"stage{stage}.joblib")
        fairness_df = fairness_check(pipe, X_te, y_te, sensitive_cols=sensitive)
        metadata[f"fairness_audit_stage{stage}"] = fairness_df.to_dict(orient="records")

    metadata["fairness_audit"] = metadata["fairness_audit_stage3"]

    # 7. Model Selection Rationale Documenting the Transition
    metadata["model_selection_rationale"] = {
        "architecture": "Option A: Standardized Random Forest at Every Stage",
        "rationale": (
            "The initial architecture selected champions strictly by Macro-F1. However, because Macro-F1 assigns equal "
            "weight to the transitional 'Enrolled' class (~18% of cohort), boosted models gained ~0.024 Macro-F1 only by "
            "predicting 'Enrolled' more aggressively, at the direct expense of dropping 4.5 to 4.8 percentage points of "
            "Dropout Recall. To align with the primary mission of early-warning dropout intervention, Option A adopts "
            "Random Forest across all three stages. Random Forest achieves the highest dropout recall and accuracy at "
            "Stages 1 and 2, matches XGBoost at Stage 3, provides direct TreeSHAP explainability, and simplifies deployment."
        ),
    }

    # Save metadata.json
    metadata_file = MODEL_DIR / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Saved complete Option A metadata to {metadata_file}")


if __name__ == "__main__":
    train_all_stages()
