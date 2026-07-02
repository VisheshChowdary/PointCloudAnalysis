from pathlib import Path
import torch


class CheckpointManager:

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)

        self.directory.mkdir(exist_ok=True)

    def save(self, model, optimizer, epoch):

        path = self.directory / f"epoch_{epoch}.pth"

        torch.save({

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict()

        }, path)

        print(f"Checkpoint saved -> {path}")