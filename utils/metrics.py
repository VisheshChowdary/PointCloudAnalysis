import torch


class Accuracy:
    """
    Computes classification accuracy.
    """

    def __call__(self, outputs, labels):

        predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == labels).sum().item()

        total = labels.size(0)

        return correct / total