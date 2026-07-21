import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class MalnutritionPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super(MalnutritionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def load_and_preprocess_data(data_path: str, num_clients: int = 3):
    """Loads the dataset, handles binary target creation, standardizes features,

    and splits data into simulated client partitions.
    """
    df = pd.read_csv(data_path)

    # WHO Standard Malnutrition Target (< -2 Z-score across stunting, wasting, or underweight)
    df['is_malnourished'] = (
        (df['stunting_zscore'] < -2.0) | 
        (df['wasting_zscore'] < -2.0) | 
        (df['underweight_zscore'] < -2.0)
    ).astype(int)

    # Exclude direct z-score target metrics to avoid target leakage
    X = df.drop(columns=[
        'is_malnourished', 'stunting_zscore', 
        'wasting_zscore', 'underweight_zscore', 'child_bmi_raw'
    ])
    y = df['is_malnourished']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Partition dataset into decentralized client splits
    X_splits = np.array_split(X_scaled, num_clients)
    y_splits = np.array_split(y.values, num_clients)

    return X_splits, y_splits, X.shape[1]