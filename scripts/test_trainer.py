import traceback

import torch
import torch.nn as nn

from configs import cfg, DATASET_ROOT
from training.dataloader import create_dataloader
from training.trainer import Trainer
from models.dummy_model import DummyModel


def main():

    try:

        print("=" * 60)
        print("PointCloudAnalysis - Phase 1")
        print("=" * 60)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Device : {device}")
        print(f"Dataset: {DATASET_ROOT}")

        train_loader = create_dataloader(
            root_dir=DATASET_ROOT,
            split="train",
            batch_size=cfg.training.batch_size,
            num_points=cfg.dataset.num_points,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers
        )

        model = DummyModel(
            num_points=cfg.dataset.num_points,
            num_classes=cfg.dataset.num_classes
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=cfg.checkpoint.directory
        )

        print("\nStarting Training...\n")

        loss, acc = trainer.train_one_epoch(train_loader)

        print("=" * 60)
        print("Training Finished")
        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f}")
        print("=" * 60)

        trainer.save(epoch=1, loss=loss)

        print("\n✓ Phase 1 Completed Successfully")

    except Exception:

        print("\nERROR OCCURRED\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()