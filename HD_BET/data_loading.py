import SimpleITK as sitk
import numpy as np
from skimage.transform import resize


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)


def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5)):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)

    assert len(image.shape) == 3, "The image has unsupported number of dimensions. Only 3D images are allowed"

    if not is_seg:
        if np.any([[i != j] for i, j in zip(spacing, spacing_target)]):
            image = resize_image(image, spacing, spacing_target).astype(np.float32)

        image -= image.mean()
        image /= image.std()
    else:
        new_shape = (int(np.round(spacing[0] / spacing_target[0] * float(image.shape[0]))),
                     int(np.round(spacing[1] / spacing_target[1] * float(image.shape[1]))),
                     int(np.round(spacing[2] / spacing_target[2] * float(image.shape[2]))))
        image = resize_segmentation(image, new_shape, 1)
    return image


def load_and_preprocess(mri_file):
    images = {}
    # t1
    images["T1"] = sitk.ReadImage(mri_file)

    properties_dict = {
        "spacing": images["T1"].GetSpacing(),
        "direction": images["T1"].GetDirection(),
        "size": images["T1"].GetSize(),
        "origin": images["T1"].GetOrigin()
    }

    for k in images.keys():
        images[k] = preprocess_image(images[k], is_seg=False, spacing_target=(1.5, 1.5, 1.5))

    properties_dict['size_before_cropping'] = images["T1"].shape

    imgs = []
    for seq in ['T1']:
        imgs.append(images[seq][None])
    all_data = np.vstack(imgs)
    print("image shape after preprocessing: ", str(all_data[0].shape))
    return all_data, properties_dict


def save_segmentation_nifti(segmentation, dct, out_fname, order=1):
    '''
    segmentation must have the same spacing as the original nifti (for now). segmentation may have been cropped out
    of the original image

    dct:
    size_before_cropping
    brain_bbox
    size -> this is the original size of the dataset, if the image was not resampled, this is the same as size_before_cropping
    spacing
    origin
    direction

    :param segmentation:
    :param dct:
    :param out_fname:
    :return:
    '''
    old_size = dct.get('size_before_cropping')
    bbox = dct.get('brain_bbox')
    if bbox is not None:
        seg_old_size = np.zeros(old_size)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + segmentation.shape[c], old_size[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
                     bbox[1][0]:bbox[1][1],
                     bbox[2][0]:bbox[2][1]] = segmentation
    else:
        seg_old_size = segmentation
    if np.any(np.array(seg_old_size) != np.array(dct['size'])[[2, 1, 0]]):
        seg_old_spacing = resize_segmentation(seg_old_size, np.array(dct['size'])[[2, 1, 0]], order=order)
    else:
        seg_old_spacing = seg_old_size
    seg_resized_itk = sitk.GetImageFromArray(seg_old_spacing.astype(np.int32))
    seg_resized_itk.SetSpacing(np.array(dct['spacing'])[[0, 1, 2]])
    seg_resized_itk.SetOrigin(dct['origin'])
    seg_resized_itk.SetDirection(dct['direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Taken from batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) to prevent dependency

    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped
