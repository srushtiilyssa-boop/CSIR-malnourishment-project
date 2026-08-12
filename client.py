"""
client.py
---------
One "site" in the federated learning demo (e.g. a clinic or health-worker
team holding its own local malnutrition screening data).

Usage:
    python client.py --client_id 1 --data data/client_1.csv

What it does:
1. Loads its own private CSV shard (never shared raw with the server).
2. Trains a local GradientBoostingClassifier to predict `malnourished`.
3. Sends its (hashed, lightly noised) model config to the federated
   server and receives back the aggregated global config.
4. Adopts the global config as its final model.
5. Scores the shared held-out test set and writes report_<id>.csv plus
   a set of diagnostic plots to the plots/ folder (saved to disk, not
   shown interactively, so this runs headless).
"""

import argparse
import hashlib
import json
import pickle
import socket

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripted/background runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "malnourished"
HOST, PORT = "localhost", 8080


def hash_column_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def reverse_hash_column_name(hashed_name: str, hash_dict: dict):
    for original, hashed in hash_dict.items():
        if hashed == hashed_name:
            return original
    return None


def add_laplace_noise(value, sensitivity, epsilon):
    return value + np.random.laplace(loc=0.0, scale=sensitivity / epsilon)


def recv_all(sock):
    BUFF_SIZE = 4096
    data = b""
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            break
    return data


def make_plots(client_id, df, X, X_train, X_test, y_test, y_pred, pred_probs, model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    importances = importances.sort_values("Importance", ascending=False)
    top5 = importances.head(5)

    # 1. Feature importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Feature", y="Importance", data=top5, hue="Feature", palette="viridis", legend=False)
    plt.title(f"Client {client_id}: Top 5 Feature Importances")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png")
    plt.close()

    # 2. Correlation heatmap of top features
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[top5["Feature"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Client {client_id}: Correlation of Top Features")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()

    # 3. Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, pred_probs, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Client {client_id}: Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve.png")
    plt.close()

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not malnourished", "Malnourished"],
                yticklabels=["Not malnourished", "Malnourished"])
    plt.title(f"Client {client_id}: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # 5. Distributions of top features by outcome
    plt.figure(figsize=(14, 8))
    for i, feature in enumerate(top5["Feature"], 1):
        plt.subplot(2, 3, i)
        sns.violinplot(x=TARGET_COL, y=feature, data=df, hue=TARGET_COL, palette="muted", legend=False)
        plt.title(feature)
    plt.tight_layout()
    plt.savefig(out_dir / "top_feature_distributions.png")
    plt.close()

    # 6. SHAP summary (sampled for speed on large shards)
    sample = X_train.sample(min(500, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png")
    plt.close()

    return importances


def client_program(client_id: int, data_path: str, test_path: str):
    plots_dir_root = f"plots/client_{client_id}"
    from pathlib import Path
    out_dir = Path(plots_dir_root)

    # --- Load this client's private local data ---
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- Train local model ---
    local_model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42
    )
    local_model.fit(X_train_scaled, y_train)
    local_val_acc = local_model.score(X_val_scaled, y_val)
    print(f"[client {client_id}] local validation accuracy: {local_val_acc:.4f}")

    # --- Federated exchange with the server ---
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    local_params = local_model.get_params()
    with open(f"outlocal_{client_id}.txt", "a") as f:
        f.write(json.dumps(local_params) + "\n")

    epsilon = 1.0
    hash_dict = {}
    encrypted_local_params = {}
    for param, value in local_params.items():
        hashed_param = hash_column_name(param)
        hash_dict[param] = hashed_param
        if isinstance(value, (int, float, np.number)):
            encrypted_local_params[hashed_param] = add_laplace_noise(value, sensitivity=20, epsilon=epsilon)
        else:
            encrypted_local_params[hashed_param] = value

    client_socket.sendall(pickle.dumps(encrypted_local_params))
    print(f"[client {client_id}] local model parameters sent to server.")

    data = recv_all(client_socket)
    if not data:
        raise ValueError("No data received from server")
    updated_global_params = pickle.loads(data)
    client_socket.close()

    decrypted_global_params = {}
    for hashed_param, value in updated_global_params.items():
        original_param = reverse_hash_column_name(hashed_param, hash_dict)
        if original_param:
            decrypted_global_params[original_param] = value

    with open(f"out_{client_id}.txt", "a") as f:
        f.write(json.dumps(decrypted_global_params) + "\n")

    # --- Adopt the federated global config as the final model ---
    final_model = GradientBoostingClassifier(**decrypted_global_params)
    final_model.fit(X_train_scaled, y_train)
    print(f"[client {client_id}] adopted global model parameters and refit locally.")

    # --- Score the shared held-out test set ---
    test_df = pd.read_csv(test_path)
    X_test_full = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    X_test_scaled = scaler.transform(X_test_full)

    y_pred = final_model.predict(X_test_scaled)
    pred_probs = final_model.predict_proba(X_test_scaled)[:, 1]

    report = test_df.copy()
    report["predicted_malnourished"] = y_pred
    report["predicted_probability"] = pred_probs
    report.to_csv(f"report_client_{client_id}.csv", index=False)
    print(f"[client {client_id}] wrote report_client_{client_id}.csv")

    make_plots(client_id, df, X, X_train, X_test_full, y_test, y_pred, pred_probs, final_model, out_dir)
    print(f"[client {client_id}] saved diagnostic plots to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to this client's local CSV shard")
    parser.add_argument("--test", type=str, default="data/test_holdout.csv", help="Path to shared held-out test CSV")
    args = parser.parse_args()
    client_program(args.client_id, args.data, args.test)
