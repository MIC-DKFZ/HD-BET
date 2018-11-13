import torch
import numpy as np


def pad_patient_3D(patient, shape_must_be_divisible_by=16, min_size=None):
    if not (isinstance(shape_must_be_divisible_by, list) or isinstance(shape_must_be_divisible_by, tuple)):
        shape_must_be_divisible_by = [shape_must_be_divisible_by] * 3
    shp = patient.shape
    new_shp = [shp[0] + shape_must_be_divisible_by[0] - shp[0] % shape_must_be_divisible_by[0],
               shp[1] + shape_must_be_divisible_by[1] - shp[1] % shape_must_be_divisible_by[1],
               shp[2] + shape_must_be_divisible_by[2] - shp[2] % shape_must_be_divisible_by[2]]
    for i in range(len(shp)):
        if shp[i] % shape_must_be_divisible_by[i] == 0:
            new_shp[i] -= shape_must_be_divisible_by[i]
    if min_size is not None:
        new_shp = np.max(np.vstack((np.array(new_shp), np.array(min_size))), 0)
    return reshape_by_padding_upper_coords(patient, new_shp, 0), shp


def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0,0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res


def predict_case_3D_net(net, patient_data, do_mirroring, num_repeats, BATCH_SIZE=None,
                           new_shape_must_be_divisible_by=16, min_size=None, main_device=0, mirror_axes=(2, 3, 4)):
    with torch.no_grad():
        pad_res = []
        for i in range(patient_data.shape[0]):
            t, old_shape = pad_patient_3D(patient_data[i], new_shape_must_be_divisible_by, min_size)
            pad_res.append(t[None])

        patient_data = np.vstack(pad_res)

        new_shp = patient_data.shape

        data = np.zeros(tuple([1] + list(new_shp)), dtype=np.float32)

        data[0] = patient_data

        if BATCH_SIZE is not None:
            data = np.vstack([data] * BATCH_SIZE)

        a = torch.rand(data.shape).float()

        if main_device == 'cpu':
            pass
        else:
            a = a.cuda(main_device)

        if do_mirroring:
            x = 8
        else:
            x = 1
        all_preds = []
        for i in range(num_repeats):
            for m in range(x):
                data_for_net = np.array(data)
                do_stuff = False
                if m == 0:
                    do_stuff = True
                    pass
                if m == 1 and (4 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, :, :, ::-1]
                if m == 2 and (3 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, :, ::-1, :]
                if m == 3 and (4 in mirror_axes) and (3 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, :, ::-1, ::-1]
                if m == 4 and (2 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, :, :]
                if m == 5 and (2 in mirror_axes) and (4 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, :, ::-1]
                if m == 6 and (2 in mirror_axes) and (3 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, ::-1, :]
                if m == 7 and (2 in mirror_axes) and (3 in mirror_axes) and (4 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, ::-1, ::-1]

                if do_stuff:
                    _ = a.data.copy_(torch.from_numpy(np.copy(data_for_net)))
                    p = net(a)  # np.copy is necessary because ::-1 creates just a view i think
                    p = p.data.cpu().numpy()

                    if m == 0:
                        pass
                    if m == 1 and (4 in mirror_axes):
                        p = p[:, :, :, :, ::-1]
                    if m == 2 and (3 in mirror_axes):
                        p = p[:, :, :, ::-1, :]
                    if m == 3 and (4 in mirror_axes) and (3 in mirror_axes):
                        p = p[:, :, :, ::-1, ::-1]
                    if m == 4 and (2 in mirror_axes):
                        p = p[:, :, ::-1, :, :]
                    if m == 5 and (2 in mirror_axes) and (4 in mirror_axes):
                        p = p[:, :, ::-1, :, ::-1]
                    if m == 6 and (2 in mirror_axes) and (3 in mirror_axes):
                        p = p[:, :, ::-1, ::-1, :]
                    if m == 7 and (2 in mirror_axes) and (3 in mirror_axes) and (4 in mirror_axes):
                        p = p[:, :, ::-1, ::-1, ::-1]
                    all_preds.append(p)

        stacked = np.vstack(all_preds)[:, :, :old_shape[0], :old_shape[1], :old_shape[2]]
        predicted_segmentation = stacked.mean(0).argmax(0)
        uncertainty = stacked.var(0)
        bayesian_predictions = stacked
        softmax_pred = stacked.mean(0)
    return predicted_segmentation, bayesian_predictions, softmax_pred, uncertainty
