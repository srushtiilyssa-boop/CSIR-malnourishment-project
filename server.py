import os
import torch
from model import load_and_preprocess_data, MalnutritionPredictor
from client import HealthCenterClient

DATA_PATH = os.path.join("dataset", "malnutrition_child_dataset_cleaned_v2.csv")
NUM_CLIENTS = 3
NUM_ROUNDS = 5

def federated_averaging(client_weights):
    """Computes FedAvg: averages model parameters across all participating clients."""
    avg_dict = {}
    for key in client_weights[0].keys():
        avg_dict[key] = sum(client_weights[i][key] for i in range(len(client_weights))) / len(client_weights)
    return avg_dict

def main():
    if not os.path.exists(DATA_PATH):
        if os.path.exists("malnutrition_child_dataset_cleaned_v2.csv"):
            data_path = "malnutrition_child_dataset_cleaned_v2.csv"
        else:
            raise FileNotFoundError(f"Dataset file not found at {DATA_PATH}.")
    else:
        data_path = DATA_PATH

    print("📊 Loading and partitioning dataset across edge nodes...")
    X_splits, y_splits, input_dim = load_and_preprocess_data(data_path, num_clients=NUM_CLIENTS)

    # Initialize Global Model
    global_model = MalnutritionPredictor(input_dim)
    
    # Initialize Local Healthcare Clients
    clients = [
        HealthCenterClient(X_splits[i], y_splits[i], input_dim) 
        for i in range(NUM_CLIENTS)
    ]

    print(f"\n🚀 Starting Native Federated Learning Loop ({NUM_CLIENTS} Healthcare Nodes, {NUM_ROUNDS} Rounds)...")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
        
        # 1. Extract global model parameters
        global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]
        
        client_weights = []
        for i, client in enumerate(clients):
            # 2. Distribute global weights to client & train locally
            updated_params, num_samples, _ = client.fit(global_params, config={})
            
            # Reconstruct updated PyTorch state_dict from client numpy arrays
            params_dict = zip(global_model.state_dict().keys(), updated_params)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            client_weights.append(state_dict)

        # 3. Aggregate client updates using FedAvg
        avg_state_dict = federated_averaging(client_weights)
        global_model.load_state_dict(avg_state_dict)

        # 4. Evaluate Global Model on Client 0's validation set
        global_params_updated = [val.cpu().numpy() for val in global_model.state_dict().values()]
        loss, _, metrics = clients[0].evaluate(global_params_updated, config={})
        print(f"Global Model Evaluation -> Loss: {loss:.4f} | Accuracy: {metrics['accuracy'] * 100:.2f}%")

    print("\n🎉 Decentralized Training Completed Successfully!")

    # Save weights inside main() scope
    torch.save(global_model.state_dict(), "global_malnutrition_model.pth")
    print("✅ Saved global model weights to 'global_malnutrition_model.pth'")

if __name__ == "__main__":
    main()