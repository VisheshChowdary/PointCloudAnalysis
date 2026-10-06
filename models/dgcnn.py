import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Compute k-nearest neighbors.

    Parameters
    ----------
    x : torch.Tensor
        Input features of shape [B, C, N]

    k : int
        Number of neighbors.

    Returns
    -------
    torch.Tensor
        Neighbor indices of shape [B, N, k]
    """

    batch_size = x.size(0)
    num_points = x.size(2)

    # Pairwise squared Euclidean distance
    inner = -2 * torch.matmul(
        x.transpose(2, 1),
        x
    )

    xx = torch.sum(
        x ** 2,
        dim=1,
        keepdim=True
    )

    pairwise_distance = (
        -xx
        - inner
        - xx.transpose(2, 1)
    )

    _, idx = pairwise_distance.topk(
        k=k,
        dim=-1
    )

    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct graph features for EdgeConv.

    Parameters
    ----------
    x : torch.Tensor
        Input features [B, C, N]

    k : int
        Number of neighbors.

    idx : torch.Tensor, optional
        Precomputed neighbor indices.

    Returns
    -------
    torch.Tensor
        Graph features [B, 2*C, N, k]
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    num_dims = x.size(1)

    if idx is None:
        idx = knn(x, k=k)

    device = x.device

    # Batch offsets
    idx_base = (
        torch.arange(
            batch_size,
            device=device
        )
        .view(-1, 1, 1)
        * num_points
    )

    idx = idx + idx_base

    idx = idx.reshape(-1)

    # [B, C, N] -> [B, N, C]
    x_transposed = x.transpose(
        2,
        1
    ).contiguous()

    neighbor_features = (
        x_transposed
        .view(
            batch_size * num_points,
            num_dims
        )[idx, :]
    )

    neighbor_features = neighbor_features.view(
        batch_size,
        num_points,
        k,
        num_dims
    )

    central_features = (
        x_transposed
        .view(
            batch_size,
            num_points,
            1,
            num_dims
        )
        .expand(
            -1,
            -1,
            k,
            -1
        )
    )

    # Edge feature:
    # [x_j - x_i, x_i]
    edge_features = torch.cat(
        (
            neighbor_features - central_features,
            central_features
        ),
        dim=3
    )

    return edge_features.permute(
        0,
        3,
        1,
        2
    ).contiguous()


class DGCNN(nn.Module):
    """
    Dynamic Graph CNN for ModelNet40 classification.

    Input:
        [B, N, 3]

    Output:
        [B, num_classes]
    """

    def __init__(
        self,
        num_classes=40,
        k=20,
        emb_dims=1024,
        dropout=0.5
    ):
        super().__init__()

        self.k = k

        # EdgeConv 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                6,
                64,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(
                negative_slope=0.2
            )
        )

        # EdgeConv 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(
                negative_slope=0.2
            )
        )

        # EdgeConv 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(
                negative_slope=0.2
            )
        )

        # EdgeConv 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(
                negative_slope=0.2
            )
        )

        # Global embedding
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                512,
                emb_dims,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(
                negative_slope=0.2
            )
        )

        # Classification head
        self.linear1 = nn.Linear(
            emb_dims * 2,
            512,
            bias=False
        )

        self.bn6 = nn.BatchNorm1d(
            512
        )

        self.dp1 = nn.Dropout(
            p=dropout
        )

        self.linear2 = nn.Linear(
            512,
            256,
            bias=False
        )

        self.bn7 = nn.BatchNorm1d(
            256
        )

        self.dp2 = nn.Dropout(
            p=dropout
        )

        self.linear3 = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [B, N, 3]

        Returns
        -------
        torch.Tensor
            [B, num_classes]
        """

        # [B, N, 3] -> [B, 3, N]
        x = x.transpose(
            2,
            1
        ).contiguous()

        # -----------------------------
        # EdgeConv 1
        # -----------------------------

        x1 = get_graph_feature(
            x,
            k=self.k
        )

        x1 = self.conv1(x1)

        x1 = x1.max(
            dim=-1,
            keepdim=False
        )[0]

        # -----------------------------
        # EdgeConv 2
        # -----------------------------

        x2 = get_graph_feature(
            x1,
            k=self.k
        )

        x2 = self.conv2(x2)

        x2 = x2.max(
            dim=-1,
            keepdim=False
        )[0]

        # -----------------------------
        # EdgeConv 3
        # -----------------------------

        x3 = get_graph_feature(
            x2,
            k=self.k
        )

        x3 = self.conv3(x3)

        x3 = x3.max(
            dim=-1,
            keepdim=False
        )[0]

        # -----------------------------
        # EdgeConv 4
        # -----------------------------

        x4 = get_graph_feature(
            x3,
            k=self.k
        )

        x4 = self.conv4(x4)

        x4 = x4.max(
            dim=-1,
            keepdim=False
        )[0]

        # -----------------------------
        # Concatenate local features
        # -----------------------------

        x = torch.cat(
            (
                x1,
                x2,
                x3,
                x4
            ),
            dim=1
        )

        # [B, 512, N]
        x = self.conv5(x)

        # -----------------------------
        # Global pooling
        # -----------------------------

        x_max = F.adaptive_max_pool1d(
            x,
            1
        ).squeeze(-1)

        x_avg = F.adaptive_avg_pool1d(
            x,
            1
        ).squeeze(-1)

        x = torch.cat(
            (
                x_max,
                x_avg
            ),
            dim=1
        )

        # -----------------------------
        # Classification
        # -----------------------------

        x = self.linear1(x)
        x = self.bn6(x)
        x = F.leaky_relu(
            x,
            negative_slope=0.2
        )
        x = self.dp1(x)

        x = self.linear2(x)
        x = self.bn7(x)
        x = F.leaky_relu(
            x,
            negative_slope=0.2
        )
        x = self.dp2(x)

        x = self.linear3(x)

        return x