import joblib
import pandas as pd
import numpy as np
from src.features import load_data
from src.train import split_data
from src.config import TARGET_COL

df = load_data()

for stage in [1, 2, 3]:
    X_train, X_test, y_train, y_test = split_data(df, stage=stage)
    pipeline = joblib.load(f"models/stage{stage}.joblib")
    
    # Predict probabilities
    probas = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)
    if "Dropout" in classes:
        dropout_idx = classes.index("Dropout")
    else:
        dropout_idx = classes.index(0) # 0 is Dropout
    dropout_probs = probas[:, dropout_idx]
    
    # Assign risk bands: Low < 0.33, Medium 0.33-0.66, High >= 0.66
    bands = []
    for p in dropout_probs:
        if p < 0.33:
            bands.append("Low")
        elif p < 0.66:
            bands.append("Medium")
        else:
            bands.append("High")
            
    is_dropout_arr = ((y_test.values == "Dropout") | (y_test.values == 0)).astype(int)
    eval_df = pd.DataFrame({
        "actual": y_test.values,
        "is_dropout": is_dropout_arr,
        "band": bands,
        "prob": dropout_probs
    })
    
    total_students = len(eval_df)
    total_dropouts = eval_df["is_dropout"].sum()
    
    print(f"\n=================== STAGE {stage} RISK BAND EVALUATION (Held-out Test, n={total_students}, Dropouts={total_dropouts}) ===================")
    for band_name in ["Low", "Medium", "High"]:
        b_df = eval_df[eval_df["band"] == band_name]
        n_b = len(b_df)
        share_students = (n_b / total_students) * 100
        n_drop = b_df["is_dropout"].sum()
        actual_dropout_rate = (n_drop / n_b * 100) if n_b > 0 else 0.0
        recall_caught = (n_drop / total_dropouts * 100) if total_dropouts > 0 else 0.0
        print(f"Risk band: {band_name:6} | Share of students: {share_students:5.1f}% (n={n_b:3}) | Actual dropout rate: {actual_dropout_rate:5.1f}% | Dropouts in band: {n_drop:3} ({recall_caught:4.1f}% of all dropouts)")
        
    # Medium + High combined
    med_high = eval_df[eval_df["band"].isin(["Medium", "High"])]
    n_mh = len(med_high)
    drop_mh = med_high["is_dropout"].sum()
    pct_caught = (drop_mh / total_dropouts) * 100
    print(f"--> Share of all true dropouts caught in Medium + High: {pct_caught:.1f}% ({drop_mh}/{total_dropouts}) | Flagged share: {(n_mh/total_students)*100:.1f}%")
