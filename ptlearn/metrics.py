# Classification metrics


def accuracy(predictions, targets):
    """ The ratio of correct predictions to ground truth for categorical data.

    Args:
        predictions (ndarray): One-hot encoded categorical prediction.
        targets (ndarray): Ground truth class labels.

    Returns:
        float: Accuracy of model.

    """
    return (predictions.argmax(axis=1) == targets).sum() / len(targets)
