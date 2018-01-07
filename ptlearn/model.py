from math import ceil
from time import time

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

from .utils import str2val, to_device
from .metrics import Accuracy, R2


OPTIM_MAP = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'SparseAdam': optim.SparseAdam,
    'Adamax': optim.Adamax,
    'ASGD': optim.ASGD,
    'LBFGS': optim.LBFGS,
    'RMSprop': optim.RMSprop,
    'Rprop': optim.Rprop,
    'SGD': optim.SGD,
}

LOSS_FN_MAP = {
    'L1': nn.L1Loss,
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss,
    'NLL': nn.NLLLoss,
    'PoissonNLL': nn.PoissonNLLLoss,
    'NLL2d': nn.NLLLoss2d,
    'KLDiv': nn.KLDivLoss,
    'BCE': nn.BCELoss,
    'BCEWithLogits': nn.BCEWithLogitsLoss,
    'MarginRanking': nn.MarginRankingLoss,
    'HingeEmbedding': nn.HingeEmbeddingLoss,
    'MultiLabelMargin': nn.MultiLabelMarginLoss,
    'SmoothL1': nn.SmoothL1Loss,
    'SoftMargin': nn.SoftMarginLoss,
    'MultiLabelSoftMargin': nn.MultiLabelSoftMarginLoss,
    'CosineEmbedding': nn.CosineEmbeddingLoss,
    'TripletMargin': nn.TripletMarginLoss,
}

METRIC_MAP = {
    'Accuracy': Accuracy,
    'R2': R2,
}


class DNN:
    """ Deep Neural Network Model.

    Args:
        net (subclass of `torch.nn.Module`): PyTorch neural network.
        loss_fn (`str` [name] or `function`): Loss function that the optimizer
            will try to minimize. Default: 'CrossEntropy'.
        optimizer (`str` [name] or subclass of `torch.optim.Optimizer`):
            Optimizer to use. Default: 'Adam'.
        metric (`str` [name] or `function`): Metric to use.
            Default: 'Accuracy'.

    """

    def __init__(self, net, loss_fn='CrossEntropy', optimizer='Adam',
                 metric='Accuracy'):
        self.net = to_device(net)
        self.loss_fn = str2val(loss_fn, LOSS_FN_MAP)()
        self.optimizer = str2val(optimizer, OPTIM_MAP)(net.parameters())
        self.metric = str2val(metric, METRIC_MAP)

    @property
    def _out_features_size(self):
        """ Get size of each output sample. """
        layer = None
        for layer in self.net.modules():
            pass
        return layer.out_features

    def _show_stats(self, epoch, total_epoch, elapse, loss, score=None):
        """ Display the metrics of model at the end of each epoch. """
        metric_str = 'Epoch {}/{} - {:.3f}s | loss: {:.3f}'.format(
            epoch, total_epoch, elapse, loss)
        if score:
            metric_str += ' | {}: {:.3g}'.format(self.metric.__name__, score)

        print(metric_str)

    def fit(self, X, Y, n_epoch=10, batch_size=128, validation_set=0.05,
            validation_batch_size=None, show_metric=True):
        """ Train the model by feeding inputs into the network and perform
        optimization.

        Args:
            X (`ndarray`): Input data for training.
            Y (`ndarray`): Targets for training.
            n_epoch (`int`): Number of full training cycles. Default: 10.
            batch_size (`int`): Number of samples to be propagated through the
                network. Default: 128.
            validation_set (`Tuple[X, Y]` or `float`, optional): Dataset for
                validation. Split training data if given a float. Don't
                validate network if argument is None. Default: 0.05.
            validation_batch_size (`int`, optional): Same as batch_size but for
                validation. If None, use batch_size. Default: None.
            show_metric (`bool`): Display metrics at every epoch if True.
                Default: True.
        Note:
            If X and/or Y is a 2d+ array each top level sub-array represents a
            sample.

        """
        if len(X) != len(Y):
            raise ValueError('X and Y differ in length.')

        if validation_set is not None:
            if isinstance(validation_set, float):
                if not (0 < validation_set < 1):
                    raise ValueError('validation_set should have range'
                                     'within (0, 1).')
                split_index = ceil(len(X) * (1 - validation_set))
                X_validate, Y_validate = X[split_index:], Y[split_index:]
                X, Y = X[0:split_index], Y[0:split_index]

            elif isinstance(validation_set, tuple) and len(validation_set) == 2:
                X_validate, Y_validate = validation_set
            else:
                raise TypeError(
                    'validation_set should have type tuple, float or None, '
                    'found: {}'.format(type(validation_set)))

        n_batches = ceil(len(X) / batch_size)

        print('Training samples: {}, validation samples: {}'.format(
            len(X), len(X_validate) if validation_set else 0))

        training_start_time = time()

        for epoch in range(n_epoch):
            epoch_loss = 0.
            epoch_start_time = time()
            # Train network
            for i in range(n_batches):
                # Calculate array index for each batch
                lo = i * batch_size
                hi = lo + batch_size

                inputs = Variable(to_device(torch.from_numpy(X[lo:hi])))
                labels = Variable(to_device(torch.from_numpy(Y[lo:hi])))

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data[0]

            # Validate network
            # XXX: This is pointless if show_metric is False.
            score = None
            if validation_set:
                score = self.evaluate(X_validate,
                                      Y_validate,
                                      validation_batch_size or batch_size)

            if show_metric:
                self._show_stats(epoch=epoch + 1,
                                 total_epoch=n_epoch,
                                 elapse=time() - epoch_start_time,
                                 loss=epoch_loss / n_batches,
                                 score=score)

        print('Training completed in {:.3f}s.'.format(
            time() - training_start_time))

    def predict(self, X, batch_size=128):
        """ Model prediction for given input data.

        Args:
            X (`ndarray`): Input data for prediction.
            batch_size (`int`): Number of samples to feed the network.
                Default: 128.

        Returns:
           `ndarray` of predicted probabilities. For classification models,
           use the `.argmax(axis=1)` method on the resulting array to obtain
           class labels.

        """
        n_batches = ceil(len(X) / batch_size)

        pred = np.empty((len(X), self._out_features_size))
        for i in range(n_batches):
            lo = i * batch_size
            hi = lo + batch_size

            inputs = Variable(to_device(torch.from_numpy(X[lo:hi])))
            pred[lo:hi] = self.net(inputs).data.cpu().numpy()

        return pred

    def evaluate(self, X, Y, batch_size=128):
        """ Evaluate model metric on given samples.

        Args:
            X (`ndarray`): Input data for evaluation.
            Y (`ndarray`): Targets for comparison with prediction.
            batch_size (`int`): Number of samples to feed the network.
                Default: 128.

        Returns:
            Metric score.

        """
        if len(X) != len(Y):
            raise ValueError('X and Y differ in length.')

        pred = self.predict(X, batch_size)

        return self.metric(pred, Y)
