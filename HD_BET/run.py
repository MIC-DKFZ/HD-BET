import torch
import numpy as np
from HD_BET.data_loading import load_and_preprocess, save_segmentation_nifti
from HD_BET.predict_case import predict_case_3D_net
import imp
from HD_BET.utils import postprocess_prediction, subfiles, maybe_mkdir_p
import os
from HD_BET.paths import folder_with_parameter_files
import HD_BET


def run(mri_fnames, output_fnames, mode, config_file=os.path.join(HD_BET.__path__[0], "config.py"), device=0,
        postprocess=False, do_tta=True):
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

        print("prediction (CNN id)...")
        for i, p in enumerate(params):
            print(i)
            net.load_state_dict(p)

            _, _, softmax_pred, _ = predict_case_3D_net(net, cf.preprocess(data), do_tta, cf.val_num_repeats,
                                                        cf.val_batch_size, cf.net_input_must_be_divisible_by,
                                                        cf.val_min_size, device, cf.da_mirror_axes)
            softmax_preds.append(softmax_pred[None])

        seg = np.argmax(np.vstack(softmax_preds).mean(0), 0)

        if postprocess:
            print("postprocessing ...")
            seg = postprocess_prediction(seg)

        print("exporting segmentation...")
        save_segmentation_nifti(seg, data_dict, out_fname)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='input. Can be either a single file name or an input folder. If file: must be '
                                       'nifti (.nii.gz) and can only be 3D. No support for 4d images, use fslsplit to '
                                       'split 4d sequences into 3d images. If folder: all files ending with .nii.gz '
                                       'within that folder will be brain extracted.', required=True, type=str)
    parser.add_argument('-output', help='output. Can be either a filename or a folder. If it does not exist, the folder'
                                     ' will be created', required=True, type=str)
    parser.add_argument('-mode', type=str, default='fast', help='can be either \'fast\' or \'accurate\'. Fast will '
                                                                'use only one set of parameters whereas accurate will '
                                                                'use the five sets of parameters that resulted from '
                                                                'our cross-validation as an ensemble. Default: fast',
                        required=False)
    parser.add_argument('-device', default='0', type=str, help='used to set on which device the prediction will run. '
                                                               'Must be either int or str. Use int for GPU id or '
                                                               '\'cpu\' to run on CPU. Default: 0',
                        required=False)
    parser.add_argument('-tta', default=1, required=False, type=int, help='whether to use test time data augmentation '
                                                                          '(mirroring). 1= True, 0=False. Disable this '
                                                                          'if you are using CPU to speed things up! '
                                                                          'Default: 1')
    parser.add_argument('-pp', default=0, type=int, required=False, help='set to 1 to enable postprocessing (remove all'
                                                                         ' but the largest connected component in '
                                                                         'the prediction. Default: 0')

    args = parser.parse_args()

    input_file_or_dir = args.input
    output_file_or_dir = args.output
    mode = args.mode
    device = args.device
    tta = args.tta
    pp = args.pp

    params_file = "model_final.model"
    config_file = "config.py"

    assert os.path.abspath(input_file_or_dir) != os.path.abspath(output_file_or_dir), "output must be different from input"

    if device == 'cpu':
        pass
    else:
        device = int(device)

    if os.path.isdir(input_file_or_dir):
        maybe_mkdir_p(output_file_or_dir)
        input_files = subfiles(input_file_or_dir, suffix='.nii.gz', join=False)

        if len(input_files) == 0:
            raise RuntimeError("input is a folder but no nifti files (.nii.gz) were found in here")

        output_files = [os.path.join(output_file_or_dir, i) for i in input_files]
        input_files = [os.path.join(input_file_or_dir, i) for i in input_files]
    else:
        if not output_file_or_dir.endswith('.nii.gz'):
            output_file_or_dir += '.nii.gz'
        output_files = [output_file_or_dir]
        input_files = [input_file_or_dir]

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unknown value for tta: %s. Expected: 0 or 1" % str(tta))

    if pp == 0:
        pp = False
    elif pp == 1:
        pp = True
    else:
        raise ValueError("Unknown value for pp: %s. Expected: 0 or 1" % str(pp))

    run(input_files, output_files, mode, config_file, device, pp, tta)
