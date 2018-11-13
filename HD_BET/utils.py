import torch
from torch import nn
import numpy as np
from skimage.morphology import label
import os


def init_weights(module):
    if isinstance(module, nn.Conv3d):
        module.weight = nn.init.kaiming_normal(module.weight, a=1e-2)
        if module.bias is not None:
            module.bias = nn.init.constant(module.bias, 0)


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class SetNetworkToVal(object):
    def __init__(self, use_dropout_sampling=False, norm_use_average=True):
        self.norm_use_average = norm_use_average
        self.use_dropout_sampling = use_dropout_sampling

    def __call__(self, module):
        if isinstance(module, nn.Dropout3d) or isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout):
            module.train(self.use_dropout_sampling)
        elif isinstance(module, nn.InstanceNorm3d) or isinstance(module, nn.InstanceNorm2d) or \
                isinstance(module, nn.InstanceNorm1d) \
                or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or \
                isinstance(module, nn.BatchNorm1d):
            module.train(not self.norm_use_average)


def postprocess_prediction(seg):
    # basically look for connected components and choose the largest one, delete everything else
    mask = seg != 0
    lbls = label(mask, 8)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


subfolders = subdirs  # I am tired of confusing those


def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            os.mkdir(os.path.join("/", *splits[:i+1]))
