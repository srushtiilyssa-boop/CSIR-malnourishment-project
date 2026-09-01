import joblib
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================
model = joblib.load("model/malnutrition_model.pkl")
scaler = joblib.load("model/scaler.pkl")
# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================
with open("model/feature_columns.json", "r") as f:
    feature_cols = json.load(f)

num_features = len(feature_cols)

print(f"Number of input features: {num_features}")
print(f"Feature columns: {feature_cols}")
# ============================================================
# DEFINE ONNX INPUT
# ============================================================
initial_type = [
    ("float_input", FloatTensorType([None, num_features]))
]
# ============================================================
# CONVERT CLASSIFIER
# ============================================================
#
# We are deliberately disabling ZipMap.
#
# For this diagnostic version, we also request ONLY the
# classifier label output.
#
# This allows us to test whether the actual classification
# inference works without the probability output.
# ============================================================
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={
        id(model): {
            "zipmap": False
        }
    },
    target_opset=17
)
# ============================================================
# REMOVE PROBABILITY OUTPUT
# ============================================================
#
# The sklearn converter normally produces:
#
#   label
#   probabilities
#
# We keep only the label output for this diagnostic test.
#
# ============================================================

label_output = onnx_model.graph.output[0]

while len(onnx_model.graph.output) > 1:
    del onnx_model.graph.output[1]
# ============================================================
# SAVE CLASSIFICATION MODEL
# ============================================================

with open("malnutrition_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Malnutrition label-only model exported successfully!")
# ============================================================
# CONVERT SCALER TO ONNX
# ============================================================
onnx_scaler = convert_sklearn(
    scaler,
    initial_types=initial_type,
    target_opset=17
)
# ============================================================
# SAVE SCALER
# ============================================================

with open("scaler.onnx", "wb") as f:
    f.write(onnx_scaler.SerializeToString())

print("✅ Scaler model exported successfully!")


print("==============================================")
print("✅ ONNX models successfully exported!")
print("==============================================")
