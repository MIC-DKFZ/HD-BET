import numpy as np
import torch
from HD_BET.utils import SetNetworkToVal, softmax_helper
from abc import abstractmethod
from HD_BET.network_architecture import Network


class BaseConfig(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_split(self, fold, random_state=12345):
        pass

    @abstractmethod
    def get_network(self, mode="train"):
        pass

    @abstractmethod
    def get_basic_generators(self, fold):
        pass

    @abstractmethod
    def get_data_generators(self, fold):
        pass

    def preprocess(self, data):
        return data

    def __repr__(self):
        res = ""
        for v in vars(self):
            if not v.startswith("__") and not v.startswith("_") and v != 'dataset':
                res += (v + ": " + str(self.__getattribute__(v)) + "\n")
        return res


class HD_BET_Config(BaseConfig):
    def __init__(self):
        super(HD_BET_Config, self).__init__()

        self.EXPERIMENT_NAME = self.__class__.__name__ # just a generic experiment name

        # network parameters
        self.net_base_num_layers = 21
        self.BATCH_SIZE = 2
        self.net_do_DS = True
        self.net_dropout_p = 0.0
        self.net_use_inst_norm = True
        self.net_conv_use_bias = True
        self.net_norm_use_affine = True
        self.net_leaky_relu_slope = 1e-1

        # hyperparameters
        self.INPUT_PATCH_SIZE = (128, 128, 128)
        self.num_classes = 2
        self.selected_data_channels = range(1)

        # data augmentation
        self.da_mirror_axes = (2, 3, 4)

        # validation
        self.val_use_DO = False
        self.val_use_train_mode = False # for dropout sampling
        self.val_num_repeats = 1 # only useful if dropout sampling
        self.val_batch_size = 1 # only useful if dropout sampling
        self.val_save_npz = True
        self.val_do_mirroring = True # test time data augmentation via mirroring
        self.val_write_images = True
        self.net_input_must_be_divisible_by = 16  # we could make a network class that has this as a property
        self.val_min_size = self.INPUT_PATCH_SIZE
        self.val_fn = None

        # CAREFUL! THIS IS A HACK TO MAKE PYTORCH 0.3 STATE DICTS COMPATIBLE WITH PYTORCH 0.4 (setting keep_runnings_
        # stats=True but not using them in validation. keep_runnings_stats was True before 0.3 but unused and defaults
        # to false in 0.4)
        self.val_use_moving_averages = False

    def get_network(self, train=True, pretrained_weights=None):
        net = Network(self.num_classes, len(self.selected_data_channels), self.net_base_num_layers,
                               self.net_dropout_p, softmax_helper, self.net_leaky_relu_slope, self.net_conv_use_bias,
                               self.net_norm_use_affine, True, self.net_do_DS)

        if pretrained_weights is not None:
            net.load_state_dict(
                torch.load(pretrained_weights, map_location=lambda storage, loc: storage))

        if train:
            net.train(True)
        else:
            net.train(False)
            net.apply(SetNetworkToVal(self.val_use_DO, self.val_use_moving_averages))
            net.do_ds = False

        optimizer = None
        self.lr_scheduler = None
        return net, optimizer

    def get_data_generators(self, fold):
        pass

    def get_split(self, fold, random_state=12345):
        pass

    def get_basic_generators(self, fold):
        pass

    def on_epoch_end(self, epoch):
        pass

    def preprocess(self, data):
        data = np.copy(data)
        for c in range(data.shape[0]):
            data[c] -= data[c].mean()
            data[c] /= data[c].std()
        return data


config = HD_BET_Config

