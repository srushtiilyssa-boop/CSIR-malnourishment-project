import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import flwr as fl

from model import MalnutritionPredictor

class HealthCenterClient(fl.client.NumPyClient):
    def __init__(self, x_data, y_data, input_dim):
        self.model = MalnutritionPredictor(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Local train/val split (80/20) at the edge node
        X_tr, X_val, y_tr, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                          torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
            batch_size=128, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                          torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)),
            batch_size=128, shuffle=False
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(2):  # 2 local epochs per round
            for x_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                preds = self.model(x_batch)
                loss += self.criterion(preds, y_batch).item()
                correct += ((preds > 0.5) == y_batch).sum().item()
        accuracy = correct / len(self.val_loader.dataset)
        return float(loss / len(self.val_loader)), len(self.val_loader.dataset), {"accuracy": float(accuracy)}