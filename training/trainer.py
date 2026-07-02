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

        self.metric = Accuracy()
        self.checkpoint = CheckpointManager(checkpoint_dir)

    def train_one_epoch(self, loader):

        print("\n==============================")
        print("Entered train_one_epoch()")
        print("==============================")

        self.model.train()

        total_loss = 0.0
        total_acc = 0.0

        for batch_idx, (points, labels) in enumerate(loader):

            print(f"\n---------- Batch {batch_idx} ----------")

            print("1. Batch Loaded")

            print("Points Shape :", points.shape)
            print("Labels Shape :", labels.shape)

            print("2. Moving to Device")

            points = points.to(self.device)
            labels = labels.to(self.device)

            print("✓ Data moved to", self.device)

            print("3. Zero Grad")

            self.optimizer.zero_grad()

            print("✓ Zero Grad Done")

            print("4. Forward Pass")

            outputs = self.model(points)

            print("✓ Forward Pass Complete")

            print("Output Shape :", outputs.shape)

            print("5. Computing Loss")

            loss = self.criterion(outputs, labels)

            print("✓ Loss :", loss.item())

            print("6. Backward Pass")

            loss.backward()

            print("✓ Backward Complete")

            print("7. Optimizer Step")

            self.optimizer.step()

            print("✓ Optimizer Step Complete")

            acc = self.metric(outputs, labels)

            print("Accuracy :", acc)

            total_loss += loss.item()
            total_acc += acc

            print("✓ Batch Finished")

            # Debug only one batch
            break

        print("\nTraining Loop Finished")

        return total_loss, total_acc

    @torch.no_grad()
    def validate(self, loader):

        print("\nValidation Started")

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

            break

        return total_loss, total_acc

    def save(self, epoch):

        print("\nSaving Checkpoint...")

        self.checkpoint.save(
            self.model,
            self.optimizer,
            epoch
        )

        print("Checkpoint Saved")