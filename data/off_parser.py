from pathlib import Path

import numpy as np
import trimesh


class OFFParser:
    """
    Loads an OFF mesh and samples points uniformly from its surface.
    """

    def __init__(self, num_points=2048):

        self.num_points = num_points

    def load(self, file_path):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        mesh = trimesh.load(file_path, force="mesh")

        points, _ = trimesh.sample.sample_surface(
            mesh,
            self.num_points
        )

        return points.astype(np.float32)