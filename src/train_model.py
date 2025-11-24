# src/train_model.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from config import PROCESSED_DIR, MODEL_DIR


def main():
    data_path = PROCESSED_DIR / "shots_for_model.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Please run extract_shots.py first."
        )

    df = pd.read_parquet(data_path)

    feature_cols = [
        "period",
        "seconds_remaining",
        "score_margin_num",
        "is_home_offense",
        "is_three",
        "is_clutch",
    ]

    # Safety: drop rows with missing features or labels
    df = df.dropna(subset=feature_cols + ["label"])
    X = df[feature_cols].astype(float).values
    y = df["label"].astype(int).values

    if len(df) < 100:
        print("Warning: very few samples, metrics may be unstable.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba)

    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"LogLoss: {ll:.4f}")

    out_path = MODEL_DIR / "lgbm_shot_model.pkl"
    joblib.dump({"model": model, "features": feature_cols}, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()