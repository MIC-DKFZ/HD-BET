import torch

from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

def main():
    print("\n########################")
    print("If you are using hd-bet, please cite the following papers:\n")
    print("Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W, "
           "Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificial "
           "neural networks. arXiv preprint arXiv:1901.11341, 2019.\n")
    print(
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.")

    print("########################\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input. Can be either a single file name or an input folder. If file: must be '
                                       'nifti (.nii.gz) and can only be 3D. No support for 4d images, use fslsplit to '
                                       'split 4d sequences into 3d images. If folder: all files ending with .nii.gz '
                                       'within that folder will be brain extracted.', required=True, type=str)
    parser.add_argument('-o', '--output', help='output. Can be either a filename or a folder. If it does not exist, the folder'
                                     ' will be created', required=False, type=str)
    parser.add_argument('-device', default='cuda', type=str, required=False,
                        help='used to set on which device the prediction will run. Can be \'cuda\' (=GPU), \'cpu\' or '
                             '\'mps\'. Default: cuda')
    parser.add_argument('--disable_tta', required=False, action='store_true',
                        help='Set this flag to disable test time augmentation. This will make prediction faster at a '
                             'slight decrease in prediction quality. Recommended for device cpu')

    parser.add_argument('--save_bet_mask', action='store_true', required=False,
                        help='Set this flag to keep the bet masks. Otherwise they will be removed once HD_BET is '
                             'done')
    parser.add_argument('--no_bet_image', action='store_true', required=False,
                        help="Set this flag to disable generating the skull stripped/brain extracted image. Only "
                             "makes sense if you also set --save_bet_mask")
    parser.add_argument('--verbose', action='store_true', required=False,
                        help="Talk to me.")

    args = parser.parse_args()

    maybe_download_parameters()

    predictor = get_hdbet_predictor(
        use_tta=not args.disable_tta,
        device=torch.device(args.device),
        verbose=args.verbose
    )

    hdbet_predict(args.input,args.output, predictor, keep_brain_mask=args.save_bet_mask,
                  compute_brain_extracted_image=not args.no_bet_image)


if __name__ == '__main__':
    main()