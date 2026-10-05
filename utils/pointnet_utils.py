import torch


def feature_transform_regularizer(trans):
    """
    Computes the PointNet feature-transform regularization loss.

    PointNet learns a feature transformation matrix T.
    Ideally, T should be close to an orthogonal matrix:

        T T^T ≈ I

    Therefore, the regularization term is:

        || T T^T - I ||_F

    Parameters
    ----------
    trans : torch.Tensor
        Feature transformation matrices.

        Expected shape:
            [B, K, K]

        where:
            B = batch size
            K = feature dimension

    Returns
    -------
    torch.Tensor
        Scalar regularization loss.
    """

    if trans is None:
        return torch.tensor(0.0)

    # ----------------------------------------------------------
    # Validate input
    # ----------------------------------------------------------

    if trans.dim() != 3:
        raise ValueError(
            "Expected transformation tensor with shape "
            "[B, K, K], but received shape: "
            f"{tuple(trans.shape)}"
        )

    batch_size = trans.size(0)
    feature_dim = trans.size(1)

    if trans.size(1) != trans.size(2):
        raise ValueError(
            "Transformation matrix must be square. "
            f"Received shape: {tuple(trans.shape)}"
        )

    # ----------------------------------------------------------
    # Create identity matrix
    # ----------------------------------------------------------

    identity = torch.eye(
        feature_dim,
        device=trans.device,
        dtype=trans.dtype
    )

    identity = identity.unsqueeze(0).expand(
        batch_size,
        -1,
        -1
    )

    # ----------------------------------------------------------
    # Compute T * T^T
    # ----------------------------------------------------------

    product = torch.bmm(
        trans,
        trans.transpose(2, 1)
    )

    # ----------------------------------------------------------
    # Orthogonality error
    #
    # T T^T should be close to I
    # ----------------------------------------------------------

    difference = product - identity

    # Frobenius norm for each matrix
    loss = torch.norm(
        difference,
        dim=(1, 2)
    )

    # Average over batch
    loss = loss.mean()

    return loss