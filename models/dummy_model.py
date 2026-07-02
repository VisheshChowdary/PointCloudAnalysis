import torch
import torch.nn as nn


class DummyModel(nn.Module):

    def __init__(
        self,
        num_points=2048,
        num_classes=40
    ):

        super().__init__()

        self.num_points = num_points

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(num_points * 3, 512),

            nn.ReLU(inplace=True),

            nn.Linear(512, 256),

            nn.ReLU(inplace=True),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        return self.network(x)