"""
train_final_model.py
---------------------
Your client.py / server.py demo trains a fresh model in memory every run and
never saves it to disk — perfect for the FL viva demo (it shows the
socket-based federation happening live), but useless for a webapp, which
needs a model file it can load instantly on every request.

This script trains ONE deployable model the same way client.py trains its
final model (same GradientBoostingClassifier hyperparameters, same
StandardScaler), fits it on all 4 client shards pooled together, and saves:
    model/malnutrition_model.pkl
    model/scaler.pkl
    model/feature_columns.json

Note on framing for your report/viva: this step is separate from the
federation mechanism itself (which only exchanges hashed, noised
hyperparameters between client.py and server.py, never raw data or model
weights). This script exists purely to produce a single deployable artifact
for the demo UI, since the socket exchange doesn't persist a model to disk.

Run this once, after data_prep.py, before webapp.py:
    python train_final_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "malnourished"
CLIENT_FILES = [f"data/client_{i}.csv" for i in range(1, 5)]
TEST_HOLDOUT = "data/test_holdout.csv"
OUT_DIR = Path("model")

# Same hyperparameters client.py starts every local model with
MODEL_PARAMS = dict(
    n_estimators=150, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42
)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    dfs = [pd.read_csv(f) for f in CLIENT_FILES]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Pooled {len(df)} rows from {len(CLIENT_FILES)} client shards")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_columns = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = GradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)

    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
    val_f1 = f1_score(y_val, model.predict(X_val_scaled))
    print(f"Validation accuracy: {val_acc:.4f}  |  F1: {val_f1:.4f}")

    # Score against the true held-out test set too, for a sanity check
    test_df = pd.read_csv(TEST_HOLDOUT)
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COL]
    X_test_scaled = scaler.transform(X_test)
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    test_f1 = f1_score(y_test, model.predict(X_test_scaled))
    print(f"Held-out test accuracy: {test_acc:.4f}  |  F1: {test_f1:.4f}")

    joblib.dump(model, OUT_DIR / "malnutrition_model.pkl")
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"\nSaved model, scaler, and feature list to {OUT_DIR}/")


if __name__ == "__main__":
    main()
