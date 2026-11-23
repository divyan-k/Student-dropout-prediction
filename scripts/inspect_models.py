import pandas as pd
from src.features import load_data
from src.train import split_data, ModelRunner
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

df = load_data()
for stage in [1, 2, 3]:
    X_train, X_test, y_train, y_test = split_data(df, stage=stage)
    print(f"\n=== STAGE {stage} ===")
    for name, clf in [
        ("RandomForest", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=3, eval_metric="mlogloss", random_state=42, n_jobs=1))
    ]:
        runner = ModelRunner(stage, clf, name)
        res = runner.cross_validate(X_train, y_train)
        print(f"{name:18} | CV Acc: {res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f} | CV Rec: {res['dropout_recall']:.4f} +/- {res['dropout_recall_std']:.4f} | CV F1: {res['macro_f1']:.4f} +/- {res['macro_f1_std']:.4f}")

