import torch
import numpy as np
import SimpleITK as sitk
from HD_BET.data_loading import load_and_preprocess, save_segmentation_nifti
from HD_BET.predict_case import predict_case_3D_net
import imp
from HD_BET.utils import postprocess_prediction, subfiles, maybe_mkdir_p
import os
from HD_BET.paths import folder_with_parameter_files
import HD_BET


def apply_bet(img, bet, out_fname):
    img_itk = sitk.ReadImage(img)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_bet = sitk.GetArrayFromImage(sitk.ReadImage(bet))
    img_npy[img_bet == 0] = 0
    out = sitk.GetImageFromArray(img_npy)
    out.CopyInformation(img_itk)
    sitk.WriteImage(out, out_fname)


def run_hd_bet(mri_fnames, output_fnames, mode="accurate", config_file=os.path.join(HD_BET.__path__[0], "config.py"), device=0,
               postprocess=False, do_tta=True, keep_mask=True):
    """

    :param mri_fnames: str or list/tuple of str
    :param output_fnames: str or list/tuple of str. If list: must have the same length as output_fnames
    :param mode: fast or accurate
    :param config_file: config.py
    :param device: either int (for device id) or 'cpu'
    :param postprocess: whether to do postprocessing or not. Postprocessing here consists of simply discarding all
    but the largest predicted connected component. Default False
    :param do_tta: whether to do test time data augmentation by mirroring along all axes. Default: True. If you use
    CPU you may want to turn that off to speed things up
    :return:
    """

    list_of_param_files = []
    if mode == 'fast':
        list_of_param_files.append(os.path.join(folder_with_parameter_files, "0.model"))
    elif mode == 'accurate':
        for i in range(5):
            list_of_param_files.append(os.path.join(folder_with_parameter_files, "%d.model" % i))
    else:
        raise ValueError("Unknown value for mode: %s. Expected: fast or accurate" % mode)
    assert all([os.path.isfile(i) for i in list_of_param_files]), "Could not find parameter files. Please refer to " \
                                                                  "the readme on how to download them"

    cf = imp.load_source('cf', config_file)
    cf = cf.config()

    net, _ = cf.get_network(cf.val_use_train_mode, None)
    if device == "cpu":
        net = net.cpu()
    else:
        net.cuda(device)

    if not isinstance(mri_fnames, (list, tuple)):
        mri_fnames = [mri_fnames]

    if not isinstance(output_fnames, (list, tuple)):
        output_fnames = [output_fnames]

    assert len(mri_fnames) == len(output_fnames), "mri_fnames and output_fnames must have the same length"

    params = []
    for p in list_of_param_files:
        params.append(torch.load(p, map_location=lambda storage, loc: storage))

    for in_fname, out_fname in zip(mri_fnames, output_fnames):
        print("File:", in_fname)
        print("preprocessing...")
        data, data_dict = load_and_preprocess(in_fname)

        softmax_preds = []

        mask_fname = out_fname[:-7] + "_mask.nii.gz"

        print("prediction (CNN id)...")
        for i, p in enumerate(params):
            print(i)
            net.load_state_dict(p)
            """assert isinstance(net, SegmentationNetwork)

            _, _, softmax_pred, _ = net.predict_3D(cf.preprocess(data), do_tta, cf.val_num_repeats, False, cf.val_batch_size,
                                                   (0, 1, 2), True, True, 2, cf.val_min_size, None, True,
                                                   'constant', {'constant_values': 0})"""
            _, _, softmax_pred, _ = predict_case_3D_net(net, cf.preprocess(data), do_tta, cf.val_num_repeats,
                                                        cf.val_batch_size, cf.net_input_must_be_divisible_by,
                                                        cf.val_min_size, device, cf.da_mirror_axes)
            softmax_preds.append(softmax_pred[None])

        seg = np.argmax(np.vstack(softmax_preds).mean(0), 0)

        if postprocess:
            print("postprocessing ...")
            seg = postprocess_prediction(seg)

        print("exporting segmentation...")
        save_segmentation_nifti(seg, data_dict, mask_fname)

        apply_bet(in_fname, mask_fname, out_fname)

        if not keep_mask:
            os.remove(mask_fname)


