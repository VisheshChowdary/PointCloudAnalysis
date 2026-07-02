from pathlib import Path

import torch

from utils.metrics import Accuracy
from utils.checkpoint import CheckpointManager


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        checkpoint_dir="checkpoints"
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric = Accuracy()
        self.checkpoint = CheckpointManager(checkpoint_dir)

    def train_one_epoch(self, loader):

        self.model.train()

        total_loss = 0.0
        total_acc = 0.0

        if len(loader) == 0:
            raise RuntimeError("Training DataLoader is empty.")

        for points, labels in loader:

            points = points.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(points)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            acc = self.metric(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

        return (
            total_loss / len(loader),
            total_acc / len(loader)
        )

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        if len(loader) == 0:
            raise RuntimeError("Validation DataLoader is empty.")

        for points, labels in loader:

            points = points.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(points)

            loss = self.criterion(outputs, labels)

            acc = self.metric(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

        return (
            total_loss / len(loader),
            total_acc / len(loader)
        )

    def save(self, epoch, loss=None):

        self.checkpoint.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss
        )