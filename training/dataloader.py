import torch

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

    if shuffle is None:
        shuffle = split == "train"

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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print("=" * 60)
    print("DataLoader Created")
    print("=" * 60)
    print("Split      :", split)
    print("Batch Size :", batch_size)
    print("Samples    :", len(dataset))
    print("Batches    :", len(loader))
    print("=" * 60)

    return loader