import torch

import ptlearn


def str2val(maybe_str, mapping):
    """ Map key to value if argument is a string."""
    if isinstance(maybe_str, str):
        return mapping[maybe_str]

    return maybe_str


def use_cuda():
    return ptlearn.use_cuda and torch.cuda.is_available()


def to_device(obj):
    """ Move neural network or tensor to the GPU if possible. """
    if use_cuda():
        return obj.cuda()

    return obj
