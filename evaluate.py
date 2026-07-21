import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from model import MalnutritionPredictor

# 1. Locate dataset
data_path = os.path.join("dataset", "malnutrition_child_dataset_cleaned_v2.csv")
if not os.path.exists(data_path):
    data_path = "malnutrition_child_dataset_cleaned_v2.csv"

# 2. Load & preprocess ground-truth dataset
df = pd.read_csv(data_path)
df['is_malnourished'] = ((df['stunting_zscore'] < -2.0) | 
                         (df['wasting_zscore'] < -2.0) | 
                         (df['underweight_zscore'] < -2.0)).astype(int)

# Drop target-derived variables to avoid data leakage
X = df.drop(columns=['is_malnourished', 'stunting_zscore', 'wasting_zscore', 'underweight_zscore', 'child_bmi_raw'])
y = df['is_malnourished'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Load saved global model
model = MalnutritionPredictor(input_dim=X.shape[1])
model.load_state_dict(torch.load("global_malnutrition_model.pth"))
model.eval()

# 4. Generate predictions
with torch.no_grad():
    inputs = torch.tensor(X_scaled, dtype=torch.float32)
    probabilities = model(inputs).numpy().squeeze()
    predictions = (probabilities > 0.5).astype(int)

# 5. Display Clinical Metrics
print("\n==================================================")
print("📊 GLOBAL MODEL EVALUATION REPORT")
print("==================================================\n")
print(classification_report(y, predictions, target_names=["Healthy", "Malnourished"]))

auc = roc_auc_score(y, probabilities)
print(f"📈 ROC-AUC Score: {auc:.4f}\n")

cm = confusion_matrix(y, predictions)
print("🧩 Confusion Matrix:")
print(f"True Positives (Correctly identified Malnourished): {cm[1][1]}")
print(f"False Negatives (Missed Malnourished cases):       {cm[1][0]}")
print(f"True Negatives (Correctly identified Healthy):     {cm[0][0]}")
print(f"False Positives (Healthy flagged as Malnourished): {cm[0][1]}")
print("==================================================\n")