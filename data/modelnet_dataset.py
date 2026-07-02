from pathlib import Path
from torch.utils.data import Dataset

from data.off_parser import OFFParser


class ModelNet40Dataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        num_points=2048,
        transform=None):

        self.transform = transform

        self.root_dir = Path(root_dir)
        self.split = split
        self.parser = OFFParser(num_points)

        self.classes = sorted(
            [folder.name for folder in self.root_dir.iterdir()
             if folder.is_dir()]
        )

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        self.samples = []

        for cls in self.classes:

            folder = self.root_dir / cls / split

            if not folder.exists():
                continue

            for file in folder.glob("*.off"):

                self.samples.append(
                    (
                        file,
                        self.class_to_idx[cls]
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        file_path, label = self.samples[index]

        points = self.parser.load(file_path)

        if self.transform:

            points = self.transform(points)

        return points, label