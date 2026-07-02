import traceback
from pathlib import Path

import torch
import torch.nn as nn

from training.dataloader import create_dataloader
from training.trainer import Trainer
from models.dummy_model import DummyModel


def find_dataset():

    possible_paths = [

        # Windows
        "datasets/ModelNet40",

        # Colab
        "/content/PointCloudAnalysis/datasets/ModelNet40",

        # Kaggle (most common)
        "/kaggle/input/modelnet40-princeton-3d-object-dataset/ModelNet40",

        # Kaggle alternative mount
        "/kaggle/input/datasets/balraj98/modelnet40-princeton-3d-object-dataset/ModelNet40",

        # Nested folder (if present)
        "/kaggle/input/modelnet40-princeton-3d-object-dataset/ModelNet40/ModelNet40",

        # Another possible nesting
        "/kaggle/input/datasets/balraj98/modelnet40-princeton-3d-object-dataset/ModelNet40/ModelNet40",
    ]

    for path in possible_paths:

        if Path(path).exists():

            print(f"\n✓ Dataset Found:\n{path}\n")

            return path

    print("\nSearching entire Kaggle input directory...\n")

    kaggle_input = Path("/kaggle/input")

    if kaggle_input.exists():

        for folder in kaggle_input.rglob("*"):

            if folder.is_dir():

                try:

                    classes = [x for x in folder.iterdir() if x.is_dir()]

                    if len(classes) >= 40:

                        print(f"\n✓ Dataset Auto Detected:\n{folder}\n")

                        return str(folder)

                except:

                    pass

    raise FileNotFoundError(
        "\nModelNet40 dataset not found.\n"
        "Please check the dataset location."
    )


def main():

    try:

        print("=" * 60)
        print("STEP 1 : Selecting Device")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Device :", device)

        print("=" * 60)
        print("STEP 2 : Finding Dataset")

        root_dir = find_dataset()

        print("=" * 60)
        print("STEP 3 : Creating DataLoader")

        train_loader = create_dataloader(

            root_dir=root_dir,

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

        print("✓ Model Created")

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

        train_loss, train_acc = trainer.train_one_epoch(train_loader)

        print("=" * 60)

        print("Training Completed")

        print(f"Loss     : {train_loss:.4f}")

        print(f"Accuracy : {train_acc:.4f}")

        trainer.save(epoch=1)

        print("\n✓ Phase 1 Completed Successfully")

    except Exception:

        print("\nERROR OCCURRED\n")

        traceback.print_exc()


if __name__ == "__main__":

    main()