import os.path
from multiprocessing import Pool

import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from HD_BETv2.paths import folder_with_parameter_files


def apply_bet(img, bet, out_fname):
    img_itk = sitk.ReadImage(img)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_bet = sitk.GetArrayFromImage(sitk.ReadImage(bet))
    img_npy[img_bet == 0] = 0
    out = sitk.GetImageFromArray(img_npy)
    out.CopyInformation(img_itk)
    sitk.WriteImage(out, out_fname)


def get_hdbet_predictor(
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        verbose: bool = False
):
    os.environ['nnUNet_compile'] = 'F'
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose
    )
    predictor.initialize_from_trained_model_folder(
        folder_with_parameter_files,
        'all'
    )
    return predictor


def hdbet_predict(
        input_file_or_folder: str,
        output_file_or_folder: str,
        predictor: nnUNetPredictor,
        keep_brain_mask: bool = False,
        compute_brain_extracted_image: bool = True
):
    # find input file or files
    if os.path.isdir((input_file_or_folder)):
        input_files = nifti_files(input_file_or_folder)
        # output_file_or_folder must be folder in this case
        maybe_mkdir_p(output_file_or_folder)
        output_files = [join(output_file_or_folder, os.path.basename(i)) for i in input_files]
        brain_mask_files = [i[:-7] + '_bet.nii.gz' for i in output_files]
    else:
        input_files = input_file_or_folder
        output_files = output_file_or_folder
        brain_mask_files = output_file_or_folder[:-7] + '_bet.nii.gz'

    # we first just predict the brain masks using the standard nnU-Net inference
    predictor.predict_from_files(
        input_files,
        brain_mask_files,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=1
    )

    if compute_brain_extracted_image:
        # now brain extract the images
        res = []
        with Pool(4) as p:
            for im, bet, out in zip(input_files, brain_mask_files, output_files):
                res.append(
                    p.starmap_async(
                    apply_bet,
                    ((im, bet, out),)
                    )
                )
            [i.get() for i in res]

    if not keep_brain_mask:
        [os.remove(i) for i in brain_mask_files]
