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


class Top_k:
    """ Top_k mean accuracy: proportion of top-k predicted labels that
    matches their corresponding target label.

    Args:
        k (`int`): The `k` in `top-k`. Default: 5.

    """
    def __init__(self, k=5):
        if not isinstance(k, int):
            raise TypeError('Top_k expects an integer.')
        self.name = 'Top_' + str(k)
        self.k = k

    def __name__(self):
        return 'Top_' + self.k

    def __call__(self, pred, targets):
        """
        Args:
            pred (`Tensor`): One-hot encoded class predictions.
            targets (`LongTensor`): Ground truth class labels.

        Returns:
            `float`. Mean accuracy.

        """
        # Topk returns (values, indices) but we only need the latter.
        matches = pred.topk(self.k, sorted=False)[1].t() == targets
        return matches.sum() / len(targets)


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


class MAE:
    """ Mean Absolute Error: measure of difference between two continuous
    variables.

    """
    def __init__(self):
        self.name = 'MAE'

    def __call__(_self, pred, targets):
        """
        Args:
            pred (`Tensor`): Predicted target values.
            targets (`Tensor`): Ground truth target values.

        Returns:
            `float`. Non-negative floating point loss, optimal value is 0.0

        """
        return (pred - targets).abs().mean()
