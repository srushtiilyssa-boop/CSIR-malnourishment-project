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

DATA = os.path.join(
    BASE, "data", "malnutrition_child_dataset_cleaned_v3 (1).csv"
)

MODEL = os.path.join(
    BASE, "model", "malnutrition_model_fixed.onnx"
)

SCALER = os.path.join(
    BASE, "model", "scaler.onnx"
)

print("=" * 60)
print("MALNUTRITION MODEL VALIDATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv(DATA)

print(f"\nDataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")

# ---------------------------------------------------------
# 2. CREATE TRUE LABEL
# ---------------------------------------------------------

df["malnourished"] = (
    (df["stunting_zscore"] < -2)
    | (df["wasting_zscore"] < -2)
    | (df["underweight_zscore"] < -2)
).astype(int)

y_true = df["malnourished"].values

# Remove columns that define the target
DROP_COLUMNS = [
    "malnourished",
    "stunting_zscore",
    "wasting_zscore",
    "underweight_zscore",
    "child_bmi_raw"
]

X = df.drop(columns=DROP_COLUMNS, errors="ignore")

print(f"Features before numeric conversion: {X.shape[1]}")

# ---------------------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------------------

# Convert categorical/text columns to numeric
X = pd.get_dummies(X, drop_first=False)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X = X.astype(np.float32)

print(f"Features supplied to model: {X.shape[1]}")

# ---------------------------------------------------------
# 4. LOAD ONNX MODEL
# ---------------------------------------------------------

print("\nLoading ONNX model...")

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

model_input = session.get_inputs()[0]
model_output = session.get_outputs()[0]

print("Model input:", model_input.name)
print("Model input shape:", model_input.shape)
print("Model output:", model_output.name)

# ---------------------------------------------------------
# 5. CHECK FEATURE COUNT
# ---------------------------------------------------------

expected_features = model_input.shape[-1]

print("\nExpected model features:", expected_features)
print("Dataset features:", X.shape[1])

if X.shape[1] != expected_features:
    raise ValueError(
        f"\nFEATURE MISMATCH!\n"
        f"Model expects {expected_features} features, "
        f"but dataset produced {X.shape[1]}."
    )

# ---------------------------------------------------------
# 6. RUN ONNX INFERENCE
# ---------------------------------------------------------

# ---------------------------------------------------------
# 6. LOAD AND APPLY SCALER
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
print("Scaler input shape:", scaler_input.shape)

print("\nApplying scaler...")

X_scaled = scaler_session.run(
    None,
    {scaler_input.name: X.values.astype(np.float32)}
)[0]

print("Scaled data shape:", X_scaled.shape)

# ---------------------------------------------------------
# 7. RUN ONNX MODEL INFERENCE
# ---------------------------------------------------------

print("\nRunning model inference...")

raw_output = session.run(
    None,
    {model_input.name: X_scaled.astype(np.float32)}
)

print("Inference completed.")

print("Inference completed.")

# ---------------------------------------------------------
# 7. EXTRACT PREDICTIONS
# ---------------------------------------------------------

print("\nRaw output information:")

for i, output in enumerate(raw_output):
    print(
        f"Output {i}: shape={np.asarray(output).shape}, "
        f"type={type(output)}"
    )

# Handle common ONNX classifier output formats
output = raw_output[0]

if isinstance(output, list):
    predictions = np.asarray(output)
else:
    predictions = np.asarray(output)

predictions = predictions.reshape(-1)

# Convert probabilities to classes if necessary
if predictions.dtype.kind in "fc":
    unique_values = np.unique(predictions)

    if not np.all(np.isin(unique_values, [0, 1])):
        predictions = (predictions >= 0.5).astype(int)

predictions = predictions.astype(int)

# ---------------------------------------------------------
# 8. VERIFY PREDICTION DISTRIBUTION
# ---------------------------------------------------------

print("\nPrediction distribution:")

unique, counts = np.unique(predictions, return_counts=True)

for value, count in zip(unique, counts):
    print(
        f"Class {value}: {count} "
        f"({count / len(predictions) * 100:.2f}%)"
    )

print("\nActual distribution:")

unique, counts = np.unique(y_true, return_counts=True)

for value, count in zip(unique, counts):
    print(
        f"Class {value}: {count} "
        f"({count / len(y_true) * 100:.2f}%)"
    )

# ---------------------------------------------------------
# 9. CALCULATE METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(
    y_true, predictions, zero_division=0
)
recall = recall_score(
    y_true, predictions, zero_division=0
)
f1 = f1_score(
    y_true, predictions, zero_division=0
)

cm = confusion_matrix(y_true, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions,
        target_names=["Nourished", "Malnourished"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 10. SAVE RESULTS
# ---------------------------------------------------------

results_dir = os.path.join(BASE, "results")
os.makedirs(results_dir, exist_ok=True)

results = pd.DataFrame({
    "Actual": y_true,
    "Predicted": predictions
})

results.to_csv(
    os.path.join(results_dir, "prediction_results.csv"),
    index=False
)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics.to_csv(
    os.path.join(results_dir, "metrics.csv"),
    index=False
)

print("\nResults saved to:")
print("results/prediction_results.csv")
print("results/metrics.csv")

print("\nVALIDATION COMPLETE.")