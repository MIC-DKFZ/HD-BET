from urllib.request import urlopen
import torch
from torch import nn
import numpy as np
from skimage.morphology import label
import os
from HD_BET.paths import folder_with_parameter_files


def get_params_fname(fold):
    return os.path.join(folder_with_parameter_files, "%d.model" % fold)


def maybe_download_parameters(fold=0, force_overwrite=False):
    """
    Downloads the parameters for some fold if it is not present yet.
    :param fold:
    :param force_overwrite: if True the old parameter file will be deleted (if present) prior to download
    :return:
    """

    assert 0 <= fold <= 4, "fold must be between 0 and 4"

    if not os.path.isdir(folder_with_parameter_files):
        maybe_mkdir_p(folder_with_parameter_files)

    out_filename = get_params_fname(fold)

    if force_overwrite and os.path.isfile(out_filename):
        os.remove(out_filename)

    if not os.path.isfile(out_filename):
        url = "https://zenodo.org/record/2540695/files/%d.model?download=1" % fold
        print("Downloading", url, "...")
        data = urlopen(url).read()
        with open(out_filename, 'wb') as f:
            f.write(data)


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
    print("running postprocessing... ")
    mask = seg != 0
    lbls = label(mask, connectivity=mask.ndim)
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
