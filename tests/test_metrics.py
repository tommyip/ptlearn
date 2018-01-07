from pytest import approx
import torch

from ptlearn import metrics


class TestClassificationMetrics:
    pred = torch.FloatTensor(
        [[0.34085051, 0.87696054, 0.88092472, 0.99508445, 0.98560877],
         [0.34837948, 0.89482408, 0.46597465, 0.82198243, 0.51986238],
         [0.62262054, 0.46592773, 0.37247103, 0.85466768, 0.51123734],
         [0.85589715, 0.18801138, 0.72825796, 0.68277607, 0.08973626],
         [0.00155825, 0.98935495, 0.43322491, 0.58208083, 0.41394411],
         [0.12851865, 0.87763327, 0.16652145, 0.39829344, 0.89590114],
         [0.52113576, 0.49376762, 0.36518342, 0.55152769, 0.97908275],
         [0.23602868, 0.35782258, 0.79911973, 0.27485965, 0.83763574],
         [0.09385917, 0.74943907, 0.77565935, 0.75978254, 0.33365698],
         [0.13145185, 0.40667302, 0.34105547, 0.75103126, 0.17661392]])

    labels = torch.LongTensor([3, 1, 2, 0, 1, 3, 4, 3, 2, 3])

    def test_accuracy(self):
        assert metrics.accuracy(self.pred, self.labels) == 0.7


class TestRegressionMetrics:
    pred = torch.FloatTensor([
        17.46309923, 21.64667588, 27.23980241, 28.77311632, 19.92267169,
        24.21549992, 22.53900011, 19.32592952, 15.08699435, 21.25263658,
        17.07986687, 21.75061173, 24.40323164, 12.83885247, 29.92750328,
        33.29491277, 15.25685785, 25.82307629, 30.36561109, 22.33368522,
        26.89140221, 12.31090742, 35.05003323, 22.85592688, 29.35431855,
        14.77870622, 12.28775257, 24.39560547, 14.95381628, 33.86956059,
        26.13222902, 25.86028662,  8.43643898, 10.49271652, 20.90849836,
        13.64741557, 22.92901396, 23.93613769, 10.74326845, 31.77288612])

    targets = torch.FloatTensor([
        15., 19., 33., 30., 16., 24., 19., 19., 15., 19., 17., 20., 25.,
        14., 32., 33., 13., 21., 29., 19., 36., 16., 37., 15., 35., 17.,
        14., 26., 18., 36., 28., 24., 10., 13., 23., 16., 25., 20., 14.,
        38.])

    def test_r2(self):
        assert metrics.r2(self.pred, self.targets) == approx(0.818234)
