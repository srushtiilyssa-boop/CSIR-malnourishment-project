"""
test/compare.py
----------------
Scores every client's post-federation report against the true labels in
the shared held-out test set, and combines all clients into a single
"federated" ensemble prediction (majority vote across clients) to show
the payoff of federation: the combined prediction should generally beat
any single client's local-only model.

Run after server.py + all client.py runs have finished:
    python test/compare.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TARGET_COL = "malnourished"
TEST_HOLDOUT = "data/test_holdout.csv"


def score_one(name, y_true, y_pred, out_dir):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not malnourished", "Malnourished"],
                yticklabels=["Not malnourished", "Malnourished"])
    plt.title(f"{name}: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)

    ground_truth = pd.read_csv(TEST_HOLDOUT)
    y_true = ground_truth[TARGET_COL].reset_index(drop=True)

    report_paths = sorted(glob.glob("report_client_*.csv"))
    if not report_paths:
        raise FileNotFoundError(
            "No report_client_*.csv files found. Run server.py and all client.py "
            "processes first (see README.md)."
        )

    results = []
    vote_frame = pd.DataFrame({"y_true": y_true})

    for path in report_paths:
        client_name = os.path.splitext(os.path.basename(path))[0]  # e.g. report_client_1
        report = pd.read_csv(path).reset_index(drop=True)
        y_pred = report["predicted_malnourished"]
        results.append(score_one(client_name, y_true, y_pred, out_dir))
        vote_frame[client_name] = y_pred

    # Federated ensemble: majority vote across all clients' post-federation models
    client_cols = [c for c in vote_frame.columns if c != "y_true"]
    vote_frame["ensemble_pred"] = (vote_frame[client_cols].mean(axis=1) >= 0.5).astype(int)
    results.append(score_one("federated_ensemble", vote_frame["y_true"], vote_frame["ensemble_pred"], out_dir))

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    print("\n=== Summary across clients + ensemble ===")
    print(summary.to_string(index=False))

    # Distribution of predicted probability of malnutrition vs ground truth, per client
    plt.figure(figsize=(10, 6))
    for path in report_paths:
        report = pd.read_csv(path)
        sns.kdeplot(report["predicted_probability"], label=os.path.basename(path))
    plt.title("Predicted probability of malnutrition, by client")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predicted_probability_distributions.png"))
    plt.close()

    print(f"\nPlots + summary_metrics.csv written to {out_dir}/")


if __name__ == "__main__":
    main()
