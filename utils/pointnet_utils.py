import torch


def feature_transform_regularizer(trans):
    """
    Orthogonality regularization used by PointNet.

    Encourages the learned feature transformation matrix
    to remain close to an orthogonal matrix.

    Parameters
    ----------
    trans : torch.Tensor
        Transformation matrices of shape:

        [B, K, K]

    Returns
    -------
    torch.Tensor
        Scalar regularization loss.
    """

    if trans is None:
        return torch.tensor(0.0)

    k = trans.size(1)

    identity = torch.eye(
        k,
        device=trans.device,
        dtype=trans.dtype
    ).unsqueeze(0)

    identity = identity.expand(
        trans.size(0),
        -1,
        -1
    )

    product = torch.bmm(
        trans,
        trans.transpose(2, 1)
    )

    diff = product - identity

    loss = torch.mean(
        torch.norm(diff, dim=(1, 2))
    )

    return loss