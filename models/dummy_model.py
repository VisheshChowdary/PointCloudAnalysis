import torch.nn as nn


class DummyModel(nn.Module):
    """
    Simple neural network used to verify that the
    complete training pipeline works correctly.

    Input:
        (B, 2048, 3)

    Output:
        (B, 40)
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(2048 * 3, 512),

            nn.ReLU(),

            nn.Linear(512, 40)

        )

    def forward(self, x):

        return self.network(x)