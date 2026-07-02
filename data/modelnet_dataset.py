from pathlib import Path
from torch.utils.data import Dataset

from data.off_parser import OFFParser


class ModelNet40Dataset(Dataset):
    """
    PyTorch Dataset for ModelNet40.

    Expected folder structure:

    datasets/
        ModelNet40/
            airplane/
                train/
                test/
            chair/
                train/
                test/
            ...
    """

    def __init__(
        self,
        root_dir,
        split="train",
        num_points=2048,
        transform=None
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.parser = OFFParser(num_points)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.root_dir}"
            )

        self.classes = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        )

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        for cls in self.classes:

            folder = self.root_dir / cls / split

            if not folder.exists():
                continue

            for file in sorted(folder.glob("*.off")):

                self.samples.append(
                    (
                        file,
                        self.class_to_idx[cls]
                    )
                )

        print("=" * 60)
        print("ModelNet40 Dataset Loaded")
        print("=" * 60)
        print(f"Split      : {self.split}")
        print(f"Classes    : {len(self.classes)}")
        print(f"Samples    : {len(self.samples)}")
        print("=" * 60)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        file_path, label = self.samples[index]

        points = self.parser.load(file_path)

        if self.transform is not None:
            points = self.transform(points)

        return points, label