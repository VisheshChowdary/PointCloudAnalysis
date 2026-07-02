from pathlib import Path
import trimesh
import numpy as np


class OFFParser:
    """
    Loads an OFF mesh and samples a fixed number of points
    uniformly from its surface.
    """

    def __init__(self, num_points=2048):
        self.num_points = num_points

    def load(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")

        mesh = trimesh.load(file_path, force="mesh")

        if mesh.is_empty:
            raise ValueError(f"Mesh {file_path} is empty.")

        points, _ = trimesh.sample.sample_surface(
            mesh,
            self.num_points
        )

        return points.astype(np.float32)