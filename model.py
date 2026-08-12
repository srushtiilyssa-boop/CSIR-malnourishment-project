import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """
    Neural network for tabular child malnutrition data.
    """

    def __init__(
        self,
        input_dim,
        num_classes
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                128
            ),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                64
            ),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


def create_model(
    input_dim,
    num_classes
):

    model = TabularMLP(
        input_dim=input_dim,
        num_classes=num_classes
    )

    return model