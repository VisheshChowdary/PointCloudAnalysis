import torch

from utils.metrics import Accuracy
from utils.checkpoint import CheckpointManager


class Trainer:
    """
    Handles training, validation and checkpointing.
    """

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

        self.metric = Accuracy()
        self.checkpoint = CheckpointManager(checkpoint_dir)

    def train_one_epoch(self, loader):

        self.model.train()

        total_loss = 0.0
        total_acc = 0.0

        for batch_idx, (points, labels) in enumerate(loader):

            points = points.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(points)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            acc = self.metric(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        for points, labels in loader:

            points = points.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(points)

            loss = self.criterion(outputs, labels)

            acc = self.metric(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)

        return avg_loss, avg_acc

    def save(self, epoch, loss=None):

        self.checkpoint.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss
        )