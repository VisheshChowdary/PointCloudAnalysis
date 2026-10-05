import torch


def square_distance(src, dst):
    """
    Calculate squared Euclidean distance.

    Parameters
    ----------
    src : torch.Tensor
        Shape [B, N, C]

    dst : torch.Tensor
        Shape [B, M, C]

    Returns
    -------
    torch.Tensor
        Shape [B, N, M]
    """

    return (
        torch.sum(src ** 2, dim=-1, keepdim=True)
        - 2 * torch.matmul(src, dst.transpose(1, 2))
        + torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    )


def index_points(points, idx):
    """
    Index points using batched indices.

    Parameters
    ----------
    points : torch.Tensor
        [B, N, C]

    idx : torch.Tensor
        [B, S] or [B, S, K]

    Returns
    -------
    torch.Tensor
        [B, S, C] or [B, S, K, C]
    """

    device = points.device

    batch_size = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = (
        torch.arange(
            batch_size,
            dtype=torch.long,
            device=device
        )
        .view(view_shape)
        .repeat(repeat_shape)
    )

    return points[batch_indices, idx]


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling.

    Parameters
    ----------
    xyz : torch.Tensor
        Point cloud [B, N, 3]

    npoint : int
        Number of points to sample.

    Returns
    -------
    torch.Tensor
        Sample indices [B, npoint]
    """

    device = xyz.device

    batch_size, num_points, _ = xyz.shape

    centroids = torch.zeros(
        batch_size,
        npoint,
        dtype=torch.long,
        device=device
    )

    distance = torch.full(
        (batch_size, num_points),
        float("inf"),
        device=device
    )

    farthest = torch.randint(
        0,
        num_points,
        (batch_size,),
        dtype=torch.long,
        device=device
    )

    batch_indices = torch.arange(
        batch_size,
        dtype=torch.long,
        device=device
    )

    for i in range(npoint):

        centroids[:, i] = farthest

        centroid = xyz[
            batch_indices,
            farthest
        ].view(batch_size, 1, 3)

        dist = torch.sum(
            (xyz - centroid) ** 2,
            dim=-1
        )

        mask = dist < distance

        distance[mask] = dist[mask]

        farthest = torch.max(
            distance,
            dim=-1
        )[1]

    return centroids


def query_ball_point(
    radius,
    nsample,
    xyz,
    new_xyz
):
    """
    Group neighboring points around sampled centroids.

    Parameters
    ----------
    radius : float
        Search radius.

    nsample : int
        Maximum number of neighbors.

    xyz : torch.Tensor
        Original points [B, N, 3]

    new_xyz : torch.Tensor
        Centroids [B, S, 3]

    Returns
    -------
    torch.Tensor
        Neighbor indices [B, S, nsample]
    """

    distances = square_distance(
        new_xyz,
        xyz
    )

    radius_squared = radius ** 2

    distances = distances.masked_fill(
        distances > radius_squared,
        float("inf")
    )

    _, group_idx = torch.topk(
        distances,
        k=min(nsample, xyz.shape[1]),
        dim=-1,
        largest=False
    )

    # If fewer than nsample valid points exist,
    # use the nearest valid point repeatedly.
    nearest_idx = group_idx[:, :, 0:1]

    invalid = torch.isinf(
        torch.gather(
            distances,
            2,
            group_idx
        )
    )

    group_idx = torch.where(
        invalid,
        nearest_idx.expand_as(group_idx),
        group_idx
    )

    return group_idx


def sample_and_group(
    npoint,
    radius,
    nsample,
    xyz,
    points
):
    """
    Perform:

        FPS
        +
        Ball Query
        +
        Local coordinate normalization
        +
        Feature grouping
    """

    fps_idx = farthest_point_sample(
        xyz,
        npoint
    )

    new_xyz = index_points(
        xyz,
        fps_idx
    )

    idx = query_ball_point(
        radius,
        nsample,
        xyz,
        new_xyz
    )

    grouped_xyz = index_points(
        xyz,
        idx
    )

    grouped_xyz = (
        grouped_xyz
        - new_xyz.unsqueeze(2)
    )

    if points is not None:

        grouped_points = index_points(
            points,
            idx
        )

        new_points = torch.cat(
            [
                grouped_xyz,
                grouped_points
            ],
            dim=-1
        )

    else:

        new_points = grouped_xyz

    return (
        new_xyz,
        new_points
    )


class PointNetSetAbstraction(torch.nn.Module):

    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel,
        mlp,
        group_all=False
    ):

        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []

        last_channel = in_channel

        for out_channel in mlp:

            layers.append(
                torch.nn.Conv2d(
                    last_channel,
                    out_channel,
                    kernel_size=1
                )
            )

            layers.append(
                torch.nn.BatchNorm2d(
                    out_channel
                )
            )

            layers.append(
                torch.nn.ReLU()
            )

            last_channel = out_channel

        self.mlp = torch.nn.Sequential(
            *layers
        )

    def forward(
        self,
        xyz,
        points
    ):
        """
        Parameters
        ----------
        xyz : [B, 3, N]

        points : [B, D, N]

        Returns
        -------
        new_xyz : [B, 3, S]

        new_points : [B, D', S]
        """

        xyz_transposed = xyz.transpose(
            1,
            2
        ).contiguous()

        if points is not None:

            points_transposed = points.transpose(
                1,
                2
            ).contiguous()

        else:

            points_transposed = None

        if self.group_all:

            new_xyz = (
                torch.mean(
                    xyz_transposed,
                    dim=1,
                    keepdim=True
                )
            )

            grouped_xyz = (
                xyz_transposed
                - new_xyz
            )

            if points_transposed is not None:

                grouped_points = (
                    points_transposed
                    .unsqueeze(1)
                )

                new_points = torch.cat(
                    [
                        grouped_xyz.unsqueeze(1),
                        grouped_points
                    ],
                    dim=-1
                )

            else:

                new_points = grouped_xyz.unsqueeze(1)

        else:

            new_xyz, new_points = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz_transposed,
                points_transposed
            )

        # [B, S, K, C]
        new_points = new_points.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        new_points = self.mlp(
            new_points
        )

        new_points = torch.max(
            new_points,
            dim=-1
        )[0]

        new_xyz = new_xyz.transpose(
            1,
            2
        ).contiguous()

        return (
            new_xyz,
            new_points
        )