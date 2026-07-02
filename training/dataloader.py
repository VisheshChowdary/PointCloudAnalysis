import torch

from torch.utils.data import DataLoader

from configs import cfg
from data.modelnet_dataset import ModelNet40Dataset
from data.transforms import (
    Compose,
    Normalize,
    RandomRotate,
    RandomJitter,
    ToTensor,
)


def create_dataloader(
    root_dir,
    split="train",
    batch_size=None,
    num_points=None,
    shuffle=None,
    num_workers=None,
):

    if batch_size is None:
        batch_size = cfg.training.batch_size

    if num_points is None:
        num_points = cfg.dataset.num_points

    if num_workers is None:
        num_workers = cfg.training.num_workers

    if shuffle is None:
        shuffle = split == "train"

    if split == "train":

        transform = Compose(
            [
                Normalize(),
                RandomRotate(),
                RandomJitter(),
                ToTensor(),
            ]
        )

    else:

        transform = Compose(
            [
                Normalize(),
                ToTensor(),
            ]
        )

    dataset = ModelNet40Dataset(
        root_dir=root_dir,
        split=split,
        num_points=num_points,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    print("=" * 60)
    print("DataLoader")
    print("=" * 60)
    print("Batch Size :", batch_size)
    print("Workers    :", num_workers)
    print("Batches    :", len(loader))
    print("=" * 60)

    return loader