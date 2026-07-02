from pathlib import Path

from torch.utils.data import Dataset

from data.off_parser import OFFParser


class ModelNet40Dataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        num_points=2048,
        transform=None
    ):

        self.transform = transform
        self.split = split
        self.parser = OFFParser(num_points)

        self.root_dir = Path(root_dir).expanduser().resolve()

        print("\n==============================")
        print("Initializing ModelNet40Dataset")
        print("==============================")
        print("Root Directory :", self.root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.root_dir}"
            )

        self.classes = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
            and (folder / "train").exists()
            and (folder / "test").exists()
        )

        print(f"Classes Found : {len(self.classes)}")

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        for cls in self.classes:

            folder = self.root_dir / cls / split

            files = sorted(folder.glob("*.off"))

            print(f"{cls:<15} {len(files)}")

            for file in files:
                self.samples.append(
                    (
                        file,
                        self.class_to_idx[cls]
                    )
                )

        print("\nTotal Samples :", len(self.samples))
        print("==============================")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        file_path, label = self.samples[idx]

        points = self.parser.load(file_path)

        if self.transform is not None:
            points = self.transform(points)

        return points, label