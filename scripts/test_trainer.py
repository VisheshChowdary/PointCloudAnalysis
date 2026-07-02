import traceback

import torch
import torch.nn as nn

from configs.paths import DATASET_ROOT
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

        print("Dataset Path :", DATASET_ROOT)

        print("=" * 60)
        print("STEP 3 : Creating DataLoader")

        train_loader = create_dataloader(
            root_dir=DATASET_ROOT,
            split="train",
            batch_size=8,
            shuffle=True,
            num_workers=2
        )

        print("✓ DataLoader Created")
        print("Dataset Size :", len(train_loader.dataset))

        print("=" * 60)
        print("STEP 4 : Creating Model")

        model = DummyModel(
            num_points=2048,
            num_classes=40
        ).to(device)

        print(model)

        print("=" * 60)
        print("STEP 5 : Creating Optimizer")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        print("✓ Optimizer Created")

        print("=" * 60)
        print("STEP 6 : Creating Loss Function")

        criterion = nn.CrossEntropyLoss()

        print("✓ Loss Function Created")

        print("=" * 60)
        print("STEP 7 : Creating Trainer")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir="checkpoints"
        )

        print("✓ Trainer Created")

        print("=" * 60)
        print("STEP 8 : Starting Training")

        loss, acc = trainer.train_one_epoch(train_loader)

        print("=" * 60)

        print("Training Finished")
        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f}")

        print("=" * 60)
        print("STEP 9 : Saving Checkpoint")

        trainer.save(epoch=1)

        print("✓ Checkpoint Saved")
        print("\n✓ Phase 1 Completed Successfully")

    except Exception:

        print("\nERROR OCCURRED\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()