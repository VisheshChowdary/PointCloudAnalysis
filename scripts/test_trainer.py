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
        print("STEP 1 : Selecting Device")

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

        print("Device :", device)

        print("=" * 60)
        print("STEP 2 : Dataset")

        print(DATASET_ROOT)

        print("=" * 60)
        print("STEP 3 : DataLoader")

        train_loader = create_dataloader(
            root_dir=DATASET_ROOT,
            split="train",
            batch_size=cfg.training.batch_size,
            num_points=cfg.dataset.num_points,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers
        )

        print("=" * 60)
        print("STEP 4 : Model")

        model = DummyModel(
            num_points=cfg.dataset.num_points,
            num_classes=cfg.dataset.num_classes
        ).to(device)

        print(model)

        print("=" * 60)
        print("STEP 5 : Optimizer")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate
        )

        print("=" * 60)
        print("STEP 6 : Loss")

        criterion = nn.CrossEntropyLoss()

        print("=" * 60)
        print("STEP 7 : Trainer")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=cfg.checkpoint.directory
        )

        print("=" * 60)
        print("STEP 8 : Training")

        loss, acc = trainer.train_one_epoch(train_loader)

        print("=" * 60)

        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f}")

        trainer.save(epoch=1, loss=loss)

        print("\n✓ Phase 1 Completed Successfully")

    except Exception:

        print("\nERROR OCCURRED\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()