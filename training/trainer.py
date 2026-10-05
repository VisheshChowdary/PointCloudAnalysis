import torch

from utils.metrics import Accuracy
from utils.checkpoint import CheckpointManager
from utils.pointnet_utils import feature_transform_regularizer


class Trainer:
    """
    Training, validation and checkpoint management.

    Supports PointNet models that return:

        outputs
        input_transform
        feature_transform

    as well as ordinary models that return only outputs.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        checkpoint_dir="checkpoints",
        feature_transform_weight=0.001
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.metric = Accuracy()

        self.checkpoint = CheckpointManager(
            checkpoint_dir
        )

        self.feature_transform_weight = (
            feature_transform_weight
        )

    def _forward(self, points):
        """
        Handles both normal classifiers and PointNet.
        """

        result = self.model(points)

        if isinstance(result, tuple):

            outputs = result[0]

            # PointNet convention:
            # result[1] = input transform
            # result[2] = feature transform

            input_transform = (
                result[1]
                if len(result) > 1
                else None
            )

            feature_transform = (
                result[2]
                if len(result) > 2
                else None
            )

            return (
                outputs,
                input_transform,
                feature_transform
            )

        return result, None, None

    def train_one_epoch(self, loader):

        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        total_correct = 0
        total_samples = 0

        for batch_idx, (points, labels) in enumerate(loader):

            points = points.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            (
                outputs,
                input_transform,
                feature_transform
            ) = self._forward(points)

            classification_loss = self.criterion(
                outputs,
                labels
            )

            if feature_transform is not None:

                reg_loss = feature_transform_regularizer(
                    feature_transform
                )

            else:

                reg_loss = torch.tensor(
                    0.0,
                    device=self.device
                )

            loss = (
                classification_loss
                +
                self.feature_transform_weight
                * reg_loss
            )

            loss.backward()

            self.optimizer.step()

            predictions = outputs.argmax(
                dim=1
            )

            correct = (
                predictions == labels
            ).sum().item()

            batch_size = labels.size(0)

            total_correct += correct
            total_samples += batch_size

            total_loss += loss.item()
            total_cls_loss += (
                classification_loss.item()
            )
            total_reg_loss += reg_loss.item()

            if (
                batch_idx + 1
            ) % 50 == 0:

                batch_acc = (
                    correct / batch_size
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {batch_acc:.4f}"
                )

        num_batches = len(loader)

        avg_loss = (
            total_loss / num_batches
        )

        avg_cls_loss = (
            total_cls_loss / num_batches
        )

        avg_reg_loss = (
            total_reg_loss / num_batches
        )

        avg_acc = (
            total_correct / total_samples
        )

        print("\nTraining Summary")
        print("-" * 60)
        print(
            f"Total Loss          : {avg_loss:.4f}"
        )
        print(
            f"Classification Loss  : {avg_cls_loss:.4f}"
        )
        print(
            f"Regularization Loss  : {avg_reg_loss:.6f}"
        )
        print(
            f"Accuracy             : {avg_acc:.4f}"
        )
        print("-" * 60)

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for points, labels in loader:

            points = points.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            (
                outputs,
                input_transform,
                feature_transform
            ) = self._forward(points)

            classification_loss = self.criterion(
                outputs,
                labels
            )

            if feature_transform is not None:

                reg_loss = feature_transform_regularizer(
                    feature_transform
                )

            else:

                reg_loss = torch.tensor(
                    0.0,
                    device=self.device
                )

            loss = (
                classification_loss
                +
                self.feature_transform_weight
                * reg_loss
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(loader)
        )

        avg_acc = (
            total_correct / total_samples
        )

        return avg_loss, avg_acc

    def save(
        self,
        epoch,
        loss=None
    ):

        self.checkpoint.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss
        )