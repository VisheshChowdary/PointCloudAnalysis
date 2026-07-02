from pathlib import Path

import torch


class CheckpointManager:
    """
    Handles saving and loading model checkpoints.
    """

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, epoch, loss=None):

        path = self.directory / f"epoch_{epoch}.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }

        torch.save(checkpoint, path)

        print("=" * 60)
        print("Checkpoint Saved")
        print(f"Epoch : {epoch}")
        print(f"Path  : {path.resolve()}")
        print("=" * 60)

    def load(self, path, model, optimizer=None, device="cpu"):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{path}"
            )

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        print("=" * 60)
        print("Checkpoint Loaded")
        print(f"Epoch : {checkpoint['epoch']}")

        if checkpoint["loss"] is not None:
            print(f"Loss  : {checkpoint['loss']:.4f}")

        print("=" * 60)

        return (
            checkpoint["epoch"],
            checkpoint.get("loss", None)
        )