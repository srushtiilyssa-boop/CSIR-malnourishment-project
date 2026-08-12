import os
import numpy as np
import torch

from model import create_model
from utils import get_data, partition_data
from client import Client
from trainer import Trainer


# ============================================================
# SETTINGS
# ============================================================

NUM_CLIENTS = 4

NUM_ROUNDS = 10

LOCAL_EPOCHS = 2

BATCH_SIZE = 128

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

    print(
        "✅ GPU detected."
    )

else:

    DEVICE = torch.device(
        "cpu"
    )

    print(
        "ℹ️ GPU not available."
    )

    print(
        "Using CPU."
    )


# ============================================================
# SET RANDOM SEED
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "DECENTRALIZED MALNUTRITION MODEL"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    train_X, train_y, test_X, test_y = (
        get_data()
    )


    # --------------------------------------------------------
    # NUMBER OF CLASSES
    # --------------------------------------------------------

    num_classes = len(
        np.unique(
            train_y
        )
    )


    # --------------------------------------------------------
    # NUMBER OF FEATURES
    # --------------------------------------------------------

    input_dim = (
        train_X.shape[1]
    )


    print(
        "\n=========================================="
    )

    print(
        "DATA INFORMATION"
    )

    print(
        "=========================================="
    )

    print(
        "Training samples:",
        len(train_X)
    )

    print(
        "Testing samples:",
        len(test_X)
    )

    print(
        "Input features:",
        input_dim
    )

    print(
        "Classes:",
        num_classes
    )


    # --------------------------------------------------------
    # CREATE SIMPLE SETTINGS OBJECT
    # --------------------------------------------------------

    class Settings:
        pass


    args = Settings()

    args.n_clients = NUM_CLIENTS

    args.n_rounds = NUM_ROUNDS

    args.n_epochs = LOCAL_EPOCHS

    args.batch_size = BATCH_SIZE

    args.lr = LEARNING_RATE

    args.device = DEVICE

    args.n_class = num_classes

    args.optimizer = "adam"

    args.momentum = 0.9

    args.verbose = 1


    # --------------------------------------------------------
    # PARTITION DATA
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "CREATING DECENTRALIZED CLIENTS"
    )

    print(
        "=========================================="
    )


    client_data = partition_data(
        train_X,
        train_y,
        args
    )


    # --------------------------------------------------------
    # CREATE CLIENT OBJECTS
    # --------------------------------------------------------

    clients = []


    for client_id in range(
        NUM_CLIENTS
    ):

        print(
            f"\nCreating Client "
            f"{client_id + 1}..."
        )


        client = Client(
            client_data[
                client_id
            ],
            args
        )


        clients.append(
            client
        )


    print(
        "\n✅ All decentralized clients created!"
    )


    # --------------------------------------------------------
    # CREATE TRAINER
    # --------------------------------------------------------

    trainer = Trainer(
        clients,
        test_X,
        test_y,
        args
    )


    # --------------------------------------------------------
    # START TRAINING
    # --------------------------------------------------------

    results = trainer.train()


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "results"
    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        output_dir,
        "decentralized_malnutrition_model.pth"
    )


    trainer.save_model(
        model_path
    )


    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history_path = os.path.join(
        output_dir,
        "training_history.npz"
    )


    np.savez(
        history_path,
        accuracy=np.array(
            results[
                "accuracy_history"
            ]
        ),
        loss=np.array(
            results[
                "loss_history"
            ]
        )
    )


    print(
        "\n✅ Training history saved!"
    )

    print(
        "File:",
        history_path
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=========================================="
    )


    final_accuracy = (
        results[
            "final_accuracy"
        ]
    )


    print(
        f"Final Accuracy: "
        f"{final_accuracy * 100:.2f}%"
    )


    print(
        f"Number of Clients: "
        f"{NUM_CLIENTS}"
    )


    print(
        f"Communication Rounds: "
        f"{NUM_ROUNDS}"
    )


    print(
        f"Local Epochs: "
        f"{LOCAL_EPOCHS}"
    )


    print(
        "\nModel saved at:"
    )

    print(
        model_path
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()