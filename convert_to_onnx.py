import joblib
import numpy as np
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load('model/malnutrition_model.pkl')
scaler = joblib.load('model/scaler.pkl')

with open('model/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

num_features = len(feature_cols)
print(f"Number of input features: {num_features}")

# 2. Define the input type for ONNX (float32 array of shape [Batch_Size, Num_Features])
initial_type = [('float_input', FloatTensorType([None, num_features]))]

# 3. Convert Scikit-Learn Model to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("malnutrition_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 4. Convert Scaler to ONNX
onnx_scaler = convert_sklearn(scaler, initial_types=initial_type)
with open("scaler.onnx", "wb") as f:
    f.write(onnx_scaler.SerializeToString())

print("✅ ONNX models successfully exported!")