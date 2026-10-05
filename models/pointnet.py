import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Transformation Network used by PointNet.

    Learns a transformation matrix that aligns either:
        - input coordinates: 3 x 3
        - intermediate features: k x k

    Input:
        x -> [B, k, N]

    Output:
        transformation matrix -> [B, k, k]
    """

    def __init__(self, k=3):
        super().__init__()

        self.k = k

        # Point-wise feature extraction
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Transformation prediction
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self._initialize_identity()

    def _initialize_identity(self):
        """
        Initialize the final transformation layer so that
        the network initially predicts an identity matrix.
        """

        nn.init.zeros_(self.fc3.weight)

        identity = torch.eye(self.k).view(-1)

        with torch.no_grad():
            self.fc3.bias.copy_(identity)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, k, N]

        Returns
        -------
        torch.Tensor
            Transformation matrix [B, k, k]
        """

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global feature
        x = torch.max(x, dim=2)[0]

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        # Predict transformation parameters
        x = self.fc3(x)

        # Reshape into matrix
        x = x.view(-1, self.k, self.k)

        return x


class PointNet(nn.Module):
    """
    PointNet classification network for ModelNet40.

    Input:
        Point cloud of shape [B, N, 3]

    Internally:
        [B, N, 3]
            ->
        [B, 3, N]

    Output:
        Class logits of shape [B, num_classes]
    """

    def __init__(
        self,
        num_classes=40,
        input_dims=3,
        use_feature_transform=True,
        dropout=0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_dims = input_dims
        self.use_feature_transform = use_feature_transform

        # --------------------------------------------------
        # Input Transformation Network
        # --------------------------------------------------

        self.input_tnet = TNet(k=input_dims)

        # --------------------------------------------------
        # First Shared MLP
        #
        # 3 -> 64 -> 64
        # --------------------------------------------------

        self.conv1 = nn.Conv1d(input_dims, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # --------------------------------------------------
        # Feature Transformation Network
        #
        # 64 x 64
        # --------------------------------------------------

        if self.use_feature_transform:
            self.feature_tnet = TNet(k=64)

        # --------------------------------------------------
        # Second Shared MLP
        #
        # 64 -> 128 -> 1024
        # --------------------------------------------------

        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)

        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        # --------------------------------------------------
        # Classification Head
        #
        # 1024 -> 512 -> 256 -> 40
        # --------------------------------------------------

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor

            Expected shape:
                [B, N, 3]

            Example:
                [32, 2048, 3]

        Returns
        -------
        logits : torch.Tensor
            Shape [B, num_classes]

        input_transform : torch.Tensor
            Shape [B, 3, 3]

        feature_transform : torch.Tensor or None
            Shape [B, 64, 64]
        """

        # --------------------------------------------------
        # Check input dimensions
        # --------------------------------------------------

        if x.ndim != 3:
            raise ValueError(
                f"Expected input with 3 dimensions [B, N, C], "
                f"but received shape {tuple(x.shape)}"
            )

        # --------------------------------------------------
        # Convert:
        #
        # [B, N, 3]
        #
        # ->
        #
        # [B, 3, N]
        #
        # Conv1d expects channels first.
        # --------------------------------------------------

        if x.shape[-1] != self.input_dims:
            raise ValueError(
                f"Expected last dimension to be {self.input_dims}, "
                f"but received shape {tuple(x.shape)}"
            )

        x = x.transpose(1, 2)

        # --------------------------------------------------
        # Input T-Net
        # --------------------------------------------------

        input_transform = self.input_tnet(x)

        # x:
        # [B, 3, N]
        #
        # transform:
        # [B, 3, 3]

        x = torch.bmm(
            input_transform,
            x
        )

        # --------------------------------------------------
        # First Shared MLP
        # --------------------------------------------------

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # --------------------------------------------------
        # Feature T-Net
        # --------------------------------------------------

        feature_transform = None

        if self.use_feature_transform:

            feature_transform = self.feature_tnet(x)

            x = torch.bmm(
                feature_transform,
                x
            )

        # --------------------------------------------------
        # Second Shared MLP
        # --------------------------------------------------

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # --------------------------------------------------
        # Global Max Pooling
        # --------------------------------------------------

        x = torch.max(
            x,
            dim=2
        )[0]

        # x:
        # [B, 1024]

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        x = F.relu(self.bn5(self.fc1(x)))

        x = self.dropout(x)

        x = F.relu(self.bn6(self.fc2(x)))

        x = self.dropout(x)

        logits = self.fc3(x)

        return (
            logits,
            input_transform,
            feature_transform
        )


def feature_transform_regularizer(transform):
    """
    Computes the PointNet feature transformation regularization.

    Encourages:

        T * T^T ≈ I

    Parameters
    ----------
    transform : torch.Tensor
        Transformation matrices.

        Shape:
            [B, K, K]

    Returns
    -------
    torch.Tensor
        Regularization loss.
    """

    if transform is None:
        return torch.tensor(0.0)

    batch_size = transform.size(0)
    k = transform.size(1)

    identity = torch.eye(
        k,
        device=transform.device,
        dtype=transform.dtype
    ).unsqueeze(0).expand(
        batch_size,
        -1,
        -1
    )

    product = torch.bmm(
        transform,
        transform.transpose(2, 1)
    )

    loss = torch.mean(
        torch.norm(
            product - identity,
            dim=(1, 2)
        )
    )

    return loss