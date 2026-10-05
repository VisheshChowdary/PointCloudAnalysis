import torch
import torch.nn as nn

from utils.pointnet2_utils import (
    PointNetSetAbstraction
)


class PointNet2(nn.Module):
    """
    PointNet++ classification model for ModelNet40.

    Input:
        [B, N, 3]

    Output:
        [B, num_classes]
    """

    def __init__(
        self,
        num_classes=40
    ):

        super().__init__()

        # -------------------------------------------------
        # Set Abstraction 1
        # -------------------------------------------------

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,
            mlp=[
                64,
                64,
                128
            ],
            group_all=False
        )

        # -------------------------------------------------
        # Set Abstraction 2
        # -------------------------------------------------

        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[
                128,
                128,
                256
            ],
            group_all=False
        )

        # -------------------------------------------------
        # Set Abstraction 3
        # -------------------------------------------------

        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[
                256,
                512,
                1024
            ],
            group_all=True
        )

        # -------------------------------------------------
        # Classification Head
        # -------------------------------------------------

        self.fc1 = nn.Linear(
            1024,
            512
        )

        self.bn1 = nn.BatchNorm1d(
            512
        )

        self.dropout1 = nn.Dropout(
            p=0.4
        )

        self.fc2 = nn.Linear(
            512,
            256
        )

        self.bn2 = nn.BatchNorm1d(
            256
        )

        self.dropout2 = nn.Dropout(
            p=0.4
        )

        self.fc3 = nn.Linear(
            256,
            num_classes
        )

    def forward(self, points):

        # Input:
        # [B, N, 3]

        xyz = points.transpose(
            1,
            2
        ).contiguous()

        # No additional point features initially
        features = None

        # SA1
        xyz, features = self.sa1(
            xyz,
            features
        )

        # SA2
        xyz, features = self.sa2(
            xyz,
            features
        )

        # SA3
        xyz, features = self.sa3(
            xyz,
            features
        )

        # [B, 1024, 1]
        features = features.squeeze(
            -1
        )

        x = self.fc1(
            features
        )

        x = self.bn1(
            x
        )

        x = torch.relu(
            x
        )

        x = self.dropout1(
            x
        )

        x = self.fc2(
            x
        )

        x = self.bn2(
            x
        )

        x = torch.relu(
            x
        )

        x = self.dropout2(
            x
        )

        logits = self.fc3(
            x
        )

        return logits