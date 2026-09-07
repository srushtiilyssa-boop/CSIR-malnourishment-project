import os
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE, "data", "test_holdout.csv")

MODEL = os.path.join(
    BASE, "model", "malnutrition_model_fixed.onnx"
)

SCALER = os.path.join(
    BASE, "model", "scaler.onnx"
)

print("=" * 65)
print("FINAL HOLDOUT VALIDATION")
print("=" * 65)

# ---------------------------------------------------------
# 1. LOAD HELD-OUT DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA)

print(f"\nHeld-out test samples: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ---------------------------------------------------------
# 2. CREATE TRUE LABEL
# ---------------------------------------------------------

df["malnourished"] = (
    (df["stunting_zscore"] < -2)
    | (df["wasting_zscore"] < -2)
    | (df["underweight_zscore"] < -2)
).astype(int)

y_true = df["malnourished"].values

# ---------------------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------------------

DROP_COLUMNS = [
    "malnourished",
    "stunting_zscore",
    "wasting_zscore",
    "underweight_zscore",
    "child_bmi_raw"
]

X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

# Convert categorical variables exactly as before
X = pd.get_dummies(
    X,
    drop_first=False
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.astype(np.float32)

print(f"Features supplied: {X.shape[1]}")

# ---------------------------------------------------------
# 4. LOAD SCALER
# ---------------------------------------------------------

print("\nLoading scaler...")

scaler_session = ort.InferenceSession(
    SCALER,
    providers=["CPUExecutionProvider"]
)

scaler_input = scaler_session.get_inputs()[0]
scaler_output = scaler_session.get_outputs()[0]

print("Scaler input:", scaler_input.name)
print("Scaler output:", scaler_output.name)

expected_features = scaler_input.shape[-1]

if X.shape[1] != expected_features:
    raise ValueError(
        f"Feature mismatch: scaler expects "
        f"{expected_features}, dataset has {X.shape[1]}"
    )

# ---------------------------------------------------------
# 5. SCALE TEST DATA
# ---------------------------------------------------------

print("\nApplying scaler...")

X_scaled = scaler_session.run(
    None,
    {
        scaler_input.name:
        X.values.astype(np.float32)
    }
)[0]

print("Scaled shape:", X_scaled.shape)

# ---------------------------------------------------------
# 6. LOAD MODEL
# ---------------------------------------------------------

print("\nLoading ONNX model...")

model_session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

model_input = model_session.get_inputs()[0]

print("Model input:", model_input.name)
print("Model input shape:", model_input.shape)

if X_scaled.shape[1] != model_input.shape[-1]:
    raise ValueError(
        f"Model expects {model_input.shape[-1]} features, "
        f"but received {X_scaled.shape[1]}"
    )

# ---------------------------------------------------------
# 7. RUN INFERENCE
# ---------------------------------------------------------

print("\nRunning inference on HELD-OUT data...")

outputs = model_session.run(
    None,
    {
        model_input.name:
        X_scaled.astype(np.float32)
    }
)

print("Inference completed.")

# ONNX classifier returns:
# outputs[0] = predicted labels
# outputs[1] = probabilities

predictions = np.asarray(outputs[0]).reshape(-1).astype(int)

# ---------------------------------------------------------
# 8. PREDICTION DISTRIBUTION
# ---------------------------------------------------------

print("\nPrediction distribution:")

for value, count in zip(
    *np.unique(predictions, return_counts=True)
):
    print(
        f"Class {value}: {count} "
        f"({count / len(predictions) * 100:.2f}%)"
    )

print("\nActual distribution:")

for value, count in zip(
    *np.unique(y_true, return_counts=True)
):
    print(
        f"Class {value}: {count} "
        f"({count / len(y_true) * 100:.2f}%)"
    )

# ---------------------------------------------------------
# 9. METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    predictions
)

# ---------------------------------------------------------
# 10. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL MODEL PERFORMANCE ON UNSEEN DATA")
print("=" * 65)

print(
    f"Accuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall    : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score  : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        predictions,
        target_names=[
            "Nourished",
            "Malnourished"
        ],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 11. SAVE RESULTS
# ---------------------------------------------------------

RESULTS = os.path.join(
    BASE,
    "results"
)

os.makedirs(
    RESULTS,
    exist_ok=True
)

prediction_results = pd.DataFrame({
    "Actual": y_true,
    "Predicted": predictions
})

prediction_results.to_csv(
    os.path.join(
        RESULTS,
        "holdout_prediction_results.csv"
    ),
    index=False
)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Test Samples"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        len(y_true)
    ]
})

metrics.to_csv(
    os.path.join(
        RESULTS,
        "holdout_metrics.csv"
    ),
    index=False
)

print("\nResults saved:")
print("results/holdout_prediction_results.csv")
print("results/holdout_metrics.csv")

print("\n" + "=" * 65)
print("FINAL HOLDOUT VALIDATION COMPLETE")
print("=" * 65)