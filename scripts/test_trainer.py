import traceback
import torch
import torch.nn as nn

from training.dataloader import create_dataloader
from training.trainer import Trainer
from models.dummy_model import DummyModel


def main():
    try:
        print("=" * 60)
        print("STEP 1 : Selecting Device")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device :", device)

        print("=" * 60)
        print("STEP 2 : Creating DataLoader")

        train_loader = create_dataloader(
            root_dir="/kaggle/input/modelnet40-princeton-3d-object-dataset/ModelNet40",
            split="train",
            batch_size=8
        )

        print("✓ DataLoader Created")

        print("=" * 60)
        print("STEP 3 : Creating Model")

        model = DummyModel().to(device)

        print(model)

        print("=" * 60)
        print("STEP 4 : Creating Optimizer")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        print("✓ Optimizer Created")

        print("=" * 60)
        print("STEP 5 : Creating Loss Function")

        criterion = nn.CrossEntropyLoss()

        print("✓ Loss Function Created")

        print("=" * 60)
        print("STEP 6 : Creating Trainer")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        print("✓ Trainer Created")

        print("=" * 60)
        print("STEP 7 : Starting Training")

        loss, acc = trainer.train_one_epoch(train_loader)

        print("=" * 60)
        print("Training Finished")

        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f}")

        print("=" * 60)
        print("STEP 8 : Saving Checkpoint")

        trainer.save(1)

        print("✓ Checkpoint Saved")

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except Exception:
        print("\nERROR OCCURRED\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()