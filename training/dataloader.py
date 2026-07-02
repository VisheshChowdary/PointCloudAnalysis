from torch.utils.data import DataLoader

from data.modelnet_dataset import ModelNet40Dataset
from data.transforms import (
    Compose,
    Normalize,
    RandomRotate,
    RandomJitter,
    ToTensor
)


def create_dataloader(
    root_dir,
    split="train",
    batch_size=32,
    num_points=2048,
    shuffle=None,
    num_workers=0
):
    """
    Creates a PyTorch DataLoader for ModelNet40.

    Parameters
    ----------
    root_dir : str
        Path to ModelNet40 dataset.

    split : str
        'train' or 'test'

    batch_size : int

    num_points : int

    shuffle : bool or None

    num_workers : int
    """

    if shuffle is None:
        shuffle = (split == "train")

    if split == "train":

        transform = Compose([
            Normalize(),
            RandomRotate(),
            RandomJitter(),
            ToTensor()
        ])

    else:

        transform = Compose([
            Normalize(),
            ToTensor()
        ])

    dataset = ModelNet40Dataset(
        root_dir=root_dir,
        split=split,
        num_points=num_points,
        transform=transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print("=" * 60)
    print("DataLoader Created")
    print("=" * 60)
    print(f"Split        : {split}")
    print(f"Batch Size   : {batch_size}")
    print(f"Samples      : {len(dataset)}")
    print(f"Batches      : {len(loader)}")
    print("=" * 60)

    return loader