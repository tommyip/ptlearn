from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

from .utils import str2val


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


class DNN:
    """ Deep Neural Network Model.

    Args:
        net (subclass of `torch.nn.Module`): PyTorch nerual network.
        loss_fn (`str` [name] or `function`): Loss function that the optimizer
            will try to minimize. Default: 'CrossEntropy'.
        optimizer (`str` [name] or subclass of `torch.optim.Optimizer`):
            Optimizer to use. Default: 'Adam'.

    """

    def __init__(self, net, loss_fn='CrossEntropy', optimizer='Adam'):
        self.net = net
        self.loss_fn = str2val(loss_fn, LOSS_FN_MAP)()
        self.optimizer = str2val(optimizer, OPTIM_MAP)(net.parameters())

    def fit(self, X, Y, n_epoch=10, batch_size=128):
        """ Train the model by feeding inputs into the network and perform
        optimization.

        Args:
            X (ndarray): Input data for training.
            Y (ndarray): Labels for training.
            n_epoch (`int`): Number of full training cycles. Default: 10.
            batch_size (`int`): Number of samples to be propagated through the
                network. Default: 128.

        Note:
            If X and/or Y is a 2d+ array each top level sub-array represents a
            sample.

        """
        n_batches = ceil(len(X) / batch_size)

        for epoch in range(n_epoch):
            epoch_loss = 0.

            for i in range(n_batches):
                # Calculate array index for each batch
                lo = i * batch_size
                hi = lo + batch_size

                inputs = Variable(torch.from_numpy(X[lo:hi]))
                labels = Variable(torch.from_numpy(Y[lo:hi]))

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data[0]

            print('Epoch: {} | loss: {}'.format(epoch, epoch_loss))

        print('PTLearn training completed.')
