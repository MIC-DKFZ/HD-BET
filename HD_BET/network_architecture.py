import torch
import torch.nn as nn
import torch.nn.functional as F
from HD_BET.utils import softmax_helper


class EncodingModule(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, dropout_p=0.3, leakiness=1e-2, conv_bias=True,
                 inst_norm_affine=True, lrelu_inplace=True):
        nn.Module.__init__(self)
        self.dropout_p = dropout_p
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.bn_1 = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, filter_size, 1, (filter_size - 1) // 2, bias=self.conv_bias)
        self.dropout = nn.Dropout3d(dropout_p)
        self.bn_2 = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, filter_size, 1, (filter_size - 1) // 2, bias=self.conv_bias)

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.bn_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.bn_2(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv2(x)
        x = x + skip
        return x


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=self.conv_bias)
        self.bn_1 = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=self.conv_bias)
        self.bn_2 = nn.InstanceNorm3d(out_channels, affine=self.inst_norm_affine, track_running_stats=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn_1(self.conv1(x)), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = F.leaky_relu(self.bn_2(self.conv2(x)), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        return x


class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.upsample = Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.upsample_conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=self.conv_bias)
        self.bn = nn.InstanceNorm3d(out_channels, affine=self.inst_norm_affine, track_running_stats=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.upsample_conv(self.upsample(x))), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        return x


class DownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.bn = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.downsample = nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=self.conv_bias)

    def forward(self, x):
        x = F.leaky_relu(self.bn(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        b = self.downsample(x)
        return x, b


class Network(nn.Module):
    def __init__(self, num_classes=4, num_input_channels=4, base_filters=16, dropout_p=0.3,
                 final_nonlin=softmax_helper, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True, do_ds=True):
        super(Network, self).__init__()

        self.do_ds = do_ds
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.final_nonlin = final_nonlin
        self.init_conv = nn.Conv3d(num_input_channels, base_filters, 3, 1, 1, bias=self.conv_bias)

        self.context1 = EncodingModule(base_filters, base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down1 = DownsamplingModule(base_filters, base_filters * 2, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context2 = EncodingModule(2 * base_filters, 2 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down2 = DownsamplingModule(2 * base_filters, base_filters * 4, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context3 = EncodingModule(4 * base_filters, 4 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down3 = DownsamplingModule(4 * base_filters, base_filters * 8, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context4 = EncodingModule(8 * base_filters, 8 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down4 = DownsamplingModule(8 * base_filters, base_filters * 16, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context5 = EncodingModule(16 * base_filters, 16 * base_filters, 3, dropout_p, leakiness=1e-2,
                                       conv_bias=True, inst_norm_affine=True, lrelu_inplace=True)

        self.bn_after_context5 = nn.InstanceNorm3d(16 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.up1 = UpsamplingModule(16 * base_filters, 8 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc1 = LocalizationModule(16 * base_filters, 8 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.up2 = UpsamplingModule(8 * base_filters, 4 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc2 = LocalizationModule(8 * base_filters, 4 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.loc2_seg = nn.Conv3d(4 * base_filters, num_classes, 1, 1, 0, bias=False)
        self.up3 = UpsamplingModule(4 * base_filters, 2 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc3 = LocalizationModule(4 * base_filters, 2 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.loc3_seg = nn.Conv3d(2 * base_filters, num_classes, 1, 1, 0, bias=False)
        self.up4 = UpsamplingModule(2 * base_filters, 1 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.end_conv_1 = nn.Conv3d(2 * base_filters, 2 * base_filters, 3, 1, 1, bias=self.conv_bias)
        self.end_conv_1_bn = nn.InstanceNorm3d(2 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.end_conv_2 = nn.Conv3d(2 * base_filters, 2 * base_filters, 3, 1, 1, bias=self.conv_bias)
        self.end_conv_2_bn = nn.InstanceNorm3d(2 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.seg_layer = nn.Conv3d(2 * base_filters, num_classes, 1, 1, 0, bias=False)

    def forward(self, x):
        seg_outputs = []

        x = self.init_conv(x)
        x = self.context1(x)

        skip1, x = self.down1(x)
        x = self.context2(x)

        skip2, x = self.down2(x)
        x = self.context3(x)

        skip3, x = self.down3(x)
        x = self.context4(x)

        skip4, x = self.down4(x)
        x = self.context5(x)

        x = F.leaky_relu(self.bn_after_context5(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.up1(x)

        x = torch.cat((skip4, x), dim=1)
        x = self.loc1(x)
        x = self.up2(x)

        x = torch.cat((skip3, x), dim=1)
        x = self.loc2(x)
        loc2_seg = self.final_nonlin(self.loc2_seg(x))
        seg_outputs.append(loc2_seg)
        x = self.up3(x)

        x = torch.cat((skip2, x), dim=1)
        x = self.loc3(x)
        loc3_seg = self.final_nonlin(self.loc3_seg(x))
        seg_outputs.append(loc3_seg)
        x = self.up4(x)

        x = torch.cat((skip1, x), dim=1)
        x = F.leaky_relu(self.end_conv_1_bn(self.end_conv_1(x)), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = F.leaky_relu(self.end_conv_2_bn(self.end_conv_2(x)), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.final_nonlin(self.seg_layer(x))
        seg_outputs.append(x)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]
