import torch

# Classification metrics


class Accuracy:
    """ Classification accuracy score: the proportion of predicted labels
    that matches exactly with their corresponding target label.

    """
    def __init__(self):
        self.name = 'Acc'

    def __call__(_self, pred, targets):
        """
        Args:
            pred (`Tensor`): One-hot encoded categorical prediction.
            targets (`LongTensor`): Ground truth class labels.

        Returns:
            float: Accuracy of model ranging [0, 1], higher is better.

        """
        # The `.max` method on a PyTorch tensor returns a tuple, the second
        # element contains the indices.
        return (pred.max(dim=1)[1] == targets).sum() / len(targets)


# Regression metrics


class R2:
    """ R^2 (Coefficient of determination) is the measure of how well the
    regression model approximates the real data samples.

    """
    def __init__(self):
        self.name = 'R2'

    def __call__(_self, pred, targets):
        """
        Args:
            pred (`Tensor`): Predicted target values.
            targets (`Tensor`): Ground truth target values.

        Returns:
            `float` (maybe be negative). A R^2 of 1 indicates that the
            regression model perfectly fits the data.

        """
        numerator = ((targets - pred) ** 2).sum(dim=0)
        # TODO: Handle divide by 0.
        denominator = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)

        return torch.mean(1 - (numerator / denominator))
