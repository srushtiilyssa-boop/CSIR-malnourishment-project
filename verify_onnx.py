import json
import joblib
import numpy as np
import onnxruntime as ort

# ==========================================
# 1. LOAD ORIGINAL PKL MODEL, SCALER & COLS
# ==========================================
pkl_model = joblib.load('model/malnutrition_model.pkl')
pkl_scaler = joblib.load('model/scaler.pkl')

with open('model/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

num_features = len(feature_cols)

# Create a sample test input row matching your feature count
# (Using fixed numbers so we can compare outputs directly)
sample_raw_input = np.ones((1, num_features), dtype=np.float32)

# ==========================================
# 2. RUN INFERENCE USING SCIKIT-LEARN (.pkl)
# ==========================================
pkl_scaled = pkl_scaler.transform(sample_raw_input)
pkl_pred = pkl_model.predict(pkl_scaled)

# Get class probabilities if supported by your model
if hasattr(pkl_model, "predict_proba"):
    pkl_prob = pkl_model.predict_proba(pkl_scaled)
else:
    pkl_prob = None

print("--- 🐍 Scikit-Learn (.pkl) Outputs ---")
print(f"Scaled Sample:     {pkl_scaled[0][:3]}...") # Printing first 3 values for brevity
print(f"Prediction Class:  {pkl_pred[0]}")
if pkl_prob is not None:
    print(f"Probabilities:     {pkl_prob[0]}")


# ==========================================
# 3. RUN INFERENCE USING ONNX RUNTIME (.onnx)
# ==========================================
scaler_session = ort.InferenceSession("scaler.onnx")
model_session = ort.InferenceSession("malnutrition_model.onnx")

# --- A. Verify Scaler ---
scaler_input_name = scaler_session.get_inputs()[0].name
onnx_scaled = scaler_session.run(None, {scaler_input_name: sample_raw_input})[0]

# --- B. Verify Model Prediction ---
model_input_name = model_session.get_inputs()[0].name
onnx_outputs = model_session.run(None, {model_input_name: onnx_scaled})

# skl2onnx outputs labels at index 0 and probabilities/dictionaries at index 1
onnx_pred = onnx_outputs[0]
onnx_prob = onnx_outputs[1] if len(onnx_outputs) > 1 else None

print("\n--- ⚡ ONNX Runtime (.onnx) Outputs ---")
print(f"Scaled Sample:     {onnx_scaled[0][:3]}...")
print(f"Prediction Class:  {onnx_pred[0]}")
if onnx_prob is not None:
    print(f"Probabilities:     {onnx_prob}")


# ==========================================
# 4. ASSERT & COMPARE EQUALITY
# ==========================================
print("\n--- 🔍 Verification Check ---")

# Check Scaler Accuracy
scaler_diff = np.max(np.abs(pkl_scaled - onnx_scaled))
print(f"Scaler Max Difference: {scaler_diff:.8f}")
assert np.allclose(pkl_scaled, onnx_scaled, atol=1e-5), "❌ Scaler outputs do not match!"

# Check Class Prediction
assert pkl_pred[0] == onnx_pred[0], "❌ Prediction labels do not match!"

print("✅ VERIFICATION SUCCESSFUL! Your ONNX model matches the PKL model perfectly.")