import numpy as np

# Classification metrics


def accuracy(pred, targets):
    """ The ratio of correct predictions to ground truth for categorical data.

    Args:
        pred (`ndarray`): One-hot encoded categorical prediction.
        targets (`ndarray`): Ground truth class labels.

    Returns:
        float: Accuracy of model ranging [0, 1], higher is better.

    """
    return (pred.argmax(axis=1) == targets).sum() / len(targets)


# Regression metrics


def r2(pred, targets):
    """ R^2 (Coefficient of determination) is the measure of how well the
    regression line approximates the real data samples.

    Args:
        pred (`ndarray`): Predicted target values.
        targets (`ndarray`): Ground truth target values.

    Returns:
        `float` (maybe be negative). A R^2 of 1 indicates that the regression
        line perfectly fits the data.

    """

    numerator = ((targets - pred) ** 2).sum(axis=0)
    # TODO: Handle divide by 0.
    denominator = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)

    return np.mean(1 - (numerator / denominator))
