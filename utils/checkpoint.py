from pathlib import Path

import torch


class CheckpointManager:

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        model,
        optimizer,
        epoch,
        loss=None
    ):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "loss": loss

        }

        path = self.directory / f"epoch_{epoch}.pth"

        torch.save(
            checkpoint,
            path
        )

        print("=" * 60)
        print("Checkpoint Saved")
        print(path)
        print("=" * 60)

    def load(
        self,
        path,
        model,
        optimizer=None,
        device="cpu"
    ):

        checkpoint = torch.load(
            path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint