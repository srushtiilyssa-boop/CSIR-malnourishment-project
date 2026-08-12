import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import create_model


class Client:
    """
    Represents one decentralized learning client.

    Each client:
    - receives its own local data
    - creates its own model
    - trains locally
    - sends model parameters to the server
    """


    # ========================================================
    # INITIALIZE CLIENT
    # ========================================================

    def __init__(
        self,
        client_data,
        args
    ):

        self.X = client_data[0]
        self.y = client_data[1]

        self.args = args

        self.device = args.device

        self.input_dim = self.X.shape[1]

        self.num_classes = args.n_class

        # Create local model
        self.model = create_model(
            input_dim=self.input_dim,
            num_classes=self.num_classes
        )

        self.model = self.model.to(
            self.device
        )


    # ========================================================
    # CREATE DATALOADER
    # ========================================================

    def get_dataloader(self):

        X_tensor = torch.tensor(
            self.X,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            self.y,
            dtype=torch.long
        )

        dataset = TensorDataset(
            X_tensor,
            y_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )

        return loader


    # ========================================================
    # SET GLOBAL MODEL
    # ========================================================

    def set_parameters(
        self,
        parameters
    ):

        state_dict = {}

        for key in parameters:

            state_dict[key] = (
                parameters[key]
                .to(self.device)
            )

        self.model.load_state_dict(
            state_dict
        )


    # ========================================================
    # GET MODEL PARAMETERS
    # ========================================================

    def get_parameters(self):

        parameters = {}

        for key, value in (
            self.model.state_dict().items()
        ):

            parameters[key] = (
                value.detach()
                .cpu()
                .clone()
            )

        return parameters


    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def train_local(
        self,
        epochs=None
    ):

        if epochs is None:

            epochs = self.args.n_epochs

        self.model.train()

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        if self.args.optimizer == "sgd":

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum
            )

        else:

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr
            )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        loader = self.get_dataloader()

        total_loss = 0.0

        total_samples = 0

        # ----------------------------------------------------
        # EPOCHS
        # ----------------------------------------------------

        for epoch in range(
            epochs
        ):

            epoch_loss = 0.0

            for batch_X, batch_y in loader:

                batch_X = batch_X.to(
                    self.device
                )

                batch_y = batch_y.to(
                    self.device
                )

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.model(
                    batch_X
                )

                # Calculate loss
                loss = criterion(
                    output,
                    batch_y
                )

                # Backpropagation
                loss.backward()

                # Update local model
                optimizer.step()

                batch_size = (
                    batch_X.size(0)
                )

                epoch_loss += (
                    loss.item()
                    * batch_size
                )

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += (
                    batch_size
                )

            average_epoch_loss = (
                epoch_loss
                / len(self.X)
            )

            if self.args.verbose:

                print(
                    f"Client local epoch "
                    f"{epoch + 1}/{epochs} "
                    f"- Loss: "
                    f"{average_epoch_loss:.4f}"
                )

        if total_samples > 0:

            average_loss = (
                total_loss
                / total_samples
            )

        else:

            average_loss = 0.0

        return average_loss


    # ========================================================
    # LOCAL EVALUATION
    # ========================================================

    def evaluate(self):

        self.model.eval()

        X_tensor = torch.tensor(
            self.X,
            dtype=torch.float32
        ).to(
            self.device
        )

        y_tensor = torch.tensor(
            self.y,
            dtype=torch.long
        ).to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
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
            self.y
        )

        if total == 0:

            accuracy = 0.0

        else:

            accuracy = (
                correct / total
            )

        return accuracy


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X
    ):

        self.model.eval()

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
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