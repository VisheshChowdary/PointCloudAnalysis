import numpy as np
import torch


class Normalize:
    """
    Center the point cloud at the origin and scale it
    to fit inside the unit sphere.
    """

    def __call__(self, points):

        centroid = np.mean(points, axis=0)

        points = points - centroid

        max_distance = np.max(np.linalg.norm(points, axis=1))

        if max_distance > 0:
            points = points / max_distance

        return points.astype(np.float32)


class RandomRotate:
    """
    Random rotation about the Y-axis.
    """

    def __call__(self, points):

        theta = np.random.uniform(0, 2 * np.pi)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ], dtype=np.float32)

        return (points @ rotation.T).astype(np.float32)


class RandomJitter:
    """
    Add Gaussian noise to each point.
    """

    def __init__(self, sigma=0.01, clip=0.02):

        self.sigma = sigma
        self.clip = clip

    def __call__(self, points):

        noise = np.clip(
            self.sigma * np.random.randn(*points.shape),
            -self.clip,
            self.clip
        )

        return (points + noise).astype(np.float32)


class ToTensor:
    """
    Convert NumPy array to PyTorch tensor.
    """

    def __call__(self, points):

        return torch.from_numpy(points).float()


class Compose:
    """
    Apply transforms sequentially.
    """

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, points):

        for transform in self.transforms:

            points = transform(points)

        return points