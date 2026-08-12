import copy
import numpy as np
import torch

from model import create_model


class Trainer:
    """
    Decentralized training coordinator.

    It:
    1. Creates the global model
    2. Sends the model to clients
    3. Collects locally trained models
    4. Aggregates their parameters
    5. Updates the global model
    """


    # ========================================================
    # INITIALIZE TRAINER
    # ========================================================

    def __init__(
        self,
        clients,
        test_X,
        test_y,
        args
    ):

        self.clients = clients

        self.test_X = test_X
        self.test_y = test_y

        self.args = args

        self.device = args.device

        # Get input size from first client
        input_dim = clients[0].input_dim

        # Create global model
        self.global_model = create_model(
            input_dim=input_dim,
            num_classes=args.n_class
        )

        self.global_model = (
            self.global_model.to(
                self.device
            )
        )


    # ========================================================
    # GET GLOBAL PARAMETERS
    # ========================================================

    def get_global_parameters(self):

        parameters = {}

        for key, value in (
            self.global_model.state_dict().items()
        ):

            parameters[key] = (
                value.detach()
                .cpu()
                .clone()
            )

        return parameters


    # ========================================================
    # SEND GLOBAL MODEL TO CLIENTS
    # ========================================================

    def distribute_model(self):

        parameters = (
            self.get_global_parameters()
        )

        for client in self.clients:

            client.set_parameters(
                parameters
            )


    # ========================================================
    # AGGREGATE CLIENT MODELS
    # ========================================================

    def aggregate(
        self,
        client_parameters,
        client_sizes
    ):

        total_samples = sum(
            client_sizes
        )

        if total_samples == 0:

            raise ValueError(
                "No client samples available."
            )

        global_state = (
            self.global_model.state_dict()
        )

        # ----------------------------------------------------
        # Weighted aggregation
        # ----------------------------------------------------

        for key in global_state:

            weighted_parameter = None

            for parameters, size in zip(
                client_parameters,
                client_sizes
            ):

                weight = (
                    size
                    / total_samples
                )

                parameter = (
                    parameters[key]
                    .to(self.device)
                )

                if weighted_parameter is None:

                    weighted_parameter = (
                        parameter
                        * weight
                    )

                else:

                    weighted_parameter += (
                        parameter
                        * weight
                    )

            global_state[key] = (
                weighted_parameter
            )

        self.global_model.load_state_dict(
            global_state
        )


    # ========================================================
    # EVALUATE GLOBAL MODEL
    # ========================================================

    def evaluate_global_model(self):

        self.global_model.eval()

        X_tensor = torch.tensor(
            self.test_X,
            dtype=torch.float32
        ).to(
            self.device
        )

        y_tensor = torch.tensor(
            self.test_y,
            dtype=torch.long
        ).to(
            self.device
        )

        with torch.no_grad():

            output = self.global_model(
                X_tensor
            )

            predictions = torch.argmax(
                output,
                dim=1
            )

        correct = (
            predictions == y_tensor
        ).sum().item()

        total = len(
            self.test_y
        )

        if total == 0:

            accuracy = 0.0

        else:

            accuracy = (
                correct / total
            )

        return accuracy


    # ========================================================
    # GET PREDICTIONS
    # ========================================================

    def predict(
        self,
        X
    ):

        self.global_model.eval()

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(
            self.device
        )

        with torch.no_grad():

            output = self.global_model(
                X_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

        return (
            predictions.cpu().numpy(),
            probabilities.cpu().numpy()
        )


    # ========================================================
    # TRAINING
    # ========================================================

    def train(self):

        print(
            "\n=========================================="
        )

        print(
            "DECENTRALIZED TRAINING"
        )

        print(
            "=========================================="
        )

        print(
            f"Number of clients: "
            f"{len(self.clients)}"
        )

        print(
            f"Communication rounds: "
            f"{self.args.n_rounds}"
        )

        print(
            f"Local epochs: "
            f"{self.args.n_epochs}"
        )


        accuracy_history = []

        loss_history = []


        # ====================================================
        # COMMUNICATION ROUNDS
        # ====================================================

        for round_number in range(
            self.args.n_rounds
        ):

            print(
                "\n------------------------------------------"
            )

            print(
                f"ROUND {round_number + 1}"
                f"/{self.args.n_rounds}"
            )

            print(
                "------------------------------------------"
            )


            # ------------------------------------------------
            # SEND GLOBAL MODEL
            # ------------------------------------------------

            self.distribute_model()


            client_parameters = []

            client_sizes = []

            client_losses = []


            # ------------------------------------------------
            # LOCAL CLIENT TRAINING
            # ------------------------------------------------

            for client_id, client in enumerate(
                self.clients
            ):

                print(
                    f"\nClient {client_id + 1}"
                    f"/{len(self.clients)}"
                )

                loss = client.train_local(
                    epochs=self.args.n_epochs
                )

                parameters = (
                    client.get_parameters()
                )

                client_parameters.append(
                    parameters
                )

                client_sizes.append(
                    len(client.X)
                )

                client_losses.append(
                    loss
                )

                print(
                    f"Local loss: {loss:.4f}"
                )


            # ------------------------------------------------
            # AGGREGATE
            # ------------------------------------------------

            self.aggregate(
                client_parameters,
                client_sizes
            )


            # ------------------------------------------------
            # GLOBAL EVALUATION
            # ------------------------------------------------

            global_accuracy = (
                self.evaluate_global_model()
            )

            average_loss = float(
                np.mean(
                    client_losses
                )
            )

            accuracy_history.append(
                global_accuracy
            )

            loss_history.append(
                average_loss
            )


            print(
                "\nGlobal model results:"
            )

            print(
                f"Average client loss: "
                f"{average_loss:.4f}"
            )

            print(
                f"Global test accuracy: "
                f"{global_accuracy * 100:.2f}%"
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        print(
            "\n=========================================="
        )

        print(
            "TRAINING FINISHED"
        )

        print(
            "=========================================="
        )

        if accuracy_history:

            final_accuracy = (
                accuracy_history[-1]
            )

            print(
                f"Final accuracy: "
                f"{final_accuracy * 100:.2f}%"
            )

        else:

            final_accuracy = 0.0


        return {
            "accuracy_history":
                accuracy_history,

            "loss_history":
                loss_history,

            "final_accuracy":
                final_accuracy
        }


    # ========================================================
    # SAVE GLOBAL MODEL
    # ========================================================

    def save_model(
        self,
        path
    ):

        torch.save(
            self.global_model.state_dict(),
            path
        )

        print(
            "\n✅ Global model saved successfully!"
        )

        print(
            "Model:",
            path
        )