#!/usr/bin/env python

import os
from HD_BET.run import run_hd_bet
from HD_BET.utils import maybe_mkdir_p, subfiles
import HD_BET


if __name__ == "__main__":
    print("\n########################")
    print("If you are using hd-bet, please cite the following paper:")
    print("Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W,"
           "Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificial"
           "neural networks. arXiv preprint arXiv:1901.11341, 2019.")
    print("########################\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input. Can be either a single file name or an input folder. If file: must be '
                                       'nifti (.nii.gz) and can only be 3D. No support for 4d images, use fslsplit to '
                                       'split 4d sequences into 3d images. If folder: all files ending with .nii.gz '
                                       'within that folder will be brain extracted.', required=True, type=str)
    parser.add_argument('-o', '--output', help='output. Can be either a filename or a folder. If it does not exist, the folder'
                                     ' will be created', required=False, type=str)
    parser.add_argument('-mode', type=str, default='accurate', help='can be either \'fast\' or \'accurate\'. Fast will '
                                                                'use only one set of parameters whereas accurate will '
                                                                'use the five sets of parameters that resulted from '
                                                                'our cross-validation as an ensemble. Default: '
                                                                    'accurate',
                        required=False)
    parser.add_argument('-device', default='0', type=str, help='used to set on which device the prediction will run. '
                                                               'Must be either int or str. Use int for GPU id or '
                                                               '\'cpu\' to run on CPU. When using CPU you should '
                                                               'consider disabling tta. Default for -device is: 0',
                        required=False)
    parser.add_argument('-tta', default=1, required=False, type=int, help='whether to use test time data augmentation '
                                                                          '(mirroring). 1= True, 0=False. Disable this '
                                                                          'if you are using CPU to speed things up! '
                                                                          'Default: 1')
    parser.add_argument('-pp', default=1, type=int, required=False, help='set to 0 to disabe postprocessing (remove all'
                                                                         ' but the largest connected component in '
                                                                         'the prediction. Default: 1')
    parser.add_argument('-s', '--save_mask', default=1, type=int, required=False, help='if set to 0 the segmentation '
                                                                                       'mask will not be '
                                                                                       'saved')
    parser.add_argument('--overwrite_existing', default=1, type=int, required=False, help="set this to 0 if you don't "
                                                                                          "want to overwrite existing "
                                                                                          "predictions")
    parser.add_argument('-b','--bet', default=1, type=int, required=False, help="set this to 0 if you don't want to save skull-stripped brain")

    args = parser.parse_args()

    input_file_or_dir = args.input
    output_file_or_dir = args.output

    if output_file_or_dir is None:
        output_file_or_dir = os.path.join(os.path.dirname(input_file_or_dir),
                                          os.path.basename(input_file_or_dir).split(".")[0] + "_bet")

    mode = args.mode
    device = args.device
    tta = args.tta
    pp = args.pp
    save_mask = args.save_mask
    overwrite_existing = args.overwrite_existing
    bet = args.bet

    params_file = os.path.join(HD_BET.__path__[0], "model_final.py")
    config_file = os.path.join(HD_BET.__path__[0], "config.py")

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
            assert os.path.abspath(input_file_or_dir) != os.path.abspath(output_file_or_dir), "output must be different from input"

        output_files = [output_file_or_dir]
        input_files = [input_file_or_dir]

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unknown value for tta: %s. Expected: 0 or 1" % str(tta))

    if overwrite_existing == 0:
        overwrite_existing = False
    elif overwrite_existing == 1:
        overwrite_existing = True
    else:
        raise ValueError("Unknown value for overwrite_existing: %s. Expected: 0 or 1" % str(overwrite_existing))

    if pp == 0:
        pp = False
    elif pp == 1:
        pp = True
    else:
        raise ValueError("Unknown value for pp: %s. Expected: 0 or 1" % str(pp))

    if save_mask == 0:
        save_mask = False
    elif save_mask == 1:
        save_mask = True
    else:
        raise ValueError("Unknown value for save_mask: %s. Expected: 0 or 1" % str(save_mask))

    if bet == 0:
        if save_mask:
            bet = False
        else:
            print("Save_mask and bet are set to 0. In this case, Bet is set to 1.")
            bet = True
    elif bet == 1:
        bet = True
    else:
        raise ValueError("Unknown value for bet: %s. Expected: 0 or 1" % str(pp))
    
    run_hd_bet(input_files, output_files, mode, config_file, device, pp, tta, save_mask, overwrite_existing, bet)
