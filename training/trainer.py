import torch

from utils.metrics import Accuracy
from utils.checkpoint import CheckpointManager
from utils.pointnet_utils import feature_transform_regularizer


class Trainer:
    """
    Training, validation and checkpoint management.

    Supports:

    1. Ordinary classification models returning:
           outputs

    2. PointNet models returning:
           outputs,
           input_transform,
           feature_transform

    PointNet feature-transform regularization is automatically
    applied when a feature transform is returned.
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

    # ==========================================================
    # FORWARD PASS
    # ==========================================================

    def _forward(self, points):
        """
        Performs model forward pass.

        Handles both:

        Ordinary models:
            outputs

        PointNet:
            outputs,
            input_transform,
            feature_transform
        """

        result = self.model(points)

        # ------------------------------------------------------
        # PointNet / tuple output
        # ------------------------------------------------------

        if isinstance(result, tuple):

            outputs = result[0]

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

        # ------------------------------------------------------
        # Normal model output
        # ------------------------------------------------------

        return result, None, None

    # ==========================================================
    # TRAIN ONE EPOCH
    # ==========================================================

    def train_one_epoch(self, loader):

        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        total_correct = 0
        total_samples = 0

        for batch_idx, (points, labels) in enumerate(loader):

            # --------------------------------------------------
            # Move data to device
            # --------------------------------------------------

            points = points.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            # --------------------------------------------------
            # Clear gradients
            # --------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            (
                outputs,
                input_transform,
                feature_transform
            ) = self._forward(points)

            # --------------------------------------------------
            # Classification loss
            # --------------------------------------------------

            classification_loss = self.criterion(
                outputs,
                labels
            )

            # --------------------------------------------------
            # PointNet feature-transform regularization
            # --------------------------------------------------

            if feature_transform is not None:

                reg_loss = feature_transform_regularizer(
                    feature_transform
                )

            else:

                reg_loss = torch.tensor(
                    0.0,
                    device=self.device
                )

            # --------------------------------------------------
            # Total loss
            # --------------------------------------------------

            loss = (
                classification_loss
                +
                self.feature_transform_weight
                * reg_loss
            )

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            loss.backward()

            self.optimizer.step()

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------

            predictions = outputs.argmax(
                dim=1
            )

            correct = (
                predictions == labels
            ).sum().item()

            batch_size = labels.size(0)

            # --------------------------------------------------
            # Accumulate statistics
            # --------------------------------------------------

            total_correct += correct
            total_samples += batch_size

            # Weight loss by batch size so that the final
            # average is a true sample-weighted average.

            total_loss += (
                loss.item() * batch_size
            )

            total_cls_loss += (
                classification_loss.item()
                * batch_size
            )

            total_reg_loss += (
                reg_loss.item()
                * batch_size
            )

            # --------------------------------------------------
            # Periodic progress
            # --------------------------------------------------

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

        # ======================================================
        # Epoch statistics
        # ======================================================

        if total_samples == 0:

            raise RuntimeError(
                "Training loader contains no samples."
            )

        avg_loss = (
            total_loss / total_samples
        )

        avg_cls_loss = (
            total_cls_loss / total_samples
        )

        avg_reg_loss = (
            total_reg_loss / total_samples
        )

        avg_acc = (
            total_correct / total_samples
        )

        # ======================================================
        # Training summary
        # ======================================================

        print()
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        print(
            f"Total Loss         : "
            f"{avg_loss:.4f}"
        )

        print(
            f"Classification Loss : "
            f"{avg_cls_loss:.4f}"
        )

        print(
            f"Regularization Loss : "
            f"{avg_reg_loss:.6f}"
        )

        print(
            f"Accuracy            : "
            f"{avg_acc:.4f}"
        )

        print(
            f"Samples             : "
            f"{total_samples}"
        )

        print("=" * 60)

        return avg_loss, avg_acc

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        total_correct = 0
        total_samples = 0

        for points, labels in loader:

            # --------------------------------------------------
            # Move data to device
            # --------------------------------------------------

            points = points.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            (
                outputs,
                input_transform,
                feature_transform
            ) = self._forward(points)

            # --------------------------------------------------
            # Classification loss
            # --------------------------------------------------

            classification_loss = self.criterion(
                outputs,
                labels
            )

            # --------------------------------------------------
            # Feature-transform regularization
            # --------------------------------------------------

            if feature_transform is not None:

                reg_loss = feature_transform_regularizer(
                    feature_transform
                )

            else:

                reg_loss = torch.tensor(
                    0.0,
                    device=self.device
                )

            # --------------------------------------------------
            # Total loss
            # --------------------------------------------------

            loss = (
                classification_loss
                +
                self.feature_transform_weight
                * reg_loss
            )

            # --------------------------------------------------
            # Predictions
            # --------------------------------------------------

            predictions = outputs.argmax(
                dim=1
            )

            correct = (
                predictions == labels
            ).sum().item()

            batch_size = labels.size(0)

            # --------------------------------------------------
            # Accumulate
            # --------------------------------------------------

            total_correct += correct
            total_samples += batch_size

            total_loss += (
                loss.item() * batch_size
            )

            total_cls_loss += (
                classification_loss.item()
                * batch_size
            )

            total_reg_loss += (
                reg_loss.item()
                * batch_size
            )

        # ======================================================
        # Validation statistics
        # ======================================================

        if total_samples == 0:

            raise RuntimeError(
                "Validation loader contains no samples."
            )

        avg_loss = (
            total_loss / total_samples
        )

        avg_cls_loss = (
            total_cls_loss / total_samples
        )

        avg_reg_loss = (
            total_reg_loss / total_samples
        )

        avg_acc = (
            total_correct / total_samples
        )

        return avg_loss, avg_acc

    # ==========================================================
    # CHECKPOINT
    # ==========================================================

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