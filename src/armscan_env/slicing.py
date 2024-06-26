import numpy as np
import SimpleITK as sitk


def padding(original_array: np.ndarray) -> np.ndarray:
    """Pad an array to make it square.

    :param original_array: array to pad
    :return: padded array.
    """
    # Find the maximum dimension
    max_dim = max(original_array.shape)

    # Calculate padding for each dimension (left and right)
    padding_x_left = (max_dim - original_array.shape[0]) // 2
    padding_x_right = max_dim - original_array.shape[0] - padding_x_left

    padding_y_left = (max_dim - original_array.shape[1]) // 2
    padding_y_right = max_dim - original_array.shape[1] - padding_y_left

    padding_z_left = (max_dim - original_array.shape[2]) // 2
    padding_z_right = max_dim - original_array.shape[2] - padding_z_left

    # Pad the array with zeros
    padded_array = np.pad(
        original_array,
        (
            (padding_x_left, padding_x_right),
            (padding_y_left, padding_y_right),
            (padding_z_left, padding_z_right),
        ),
        mode="constant",
    )

    # Verify the shapes
    print("Original Array Shape:", original_array.shape)
    print("Padded Array Shape:", padded_array.shape)

    return padded_array


def transform_volume(
    volume: sitk.Image,
    z_rotation: float | np.ndarray = 0.0,
    x_rotation: float | np.ndarray = 0.0,
    x_trans: float | np.ndarray = 0.0,
    y_trans: float | np.ndarray = 0.0,
) -> sitk.Image:
    """Trasnform a 3D volume with arbitrary rotation and translation.

    :param z_rotation: rotation around z-axis in degrees
    :param x_rotation: rotation around x-axis in degrees
    :param x_trans: translation along x-axis
    :param y_trans: translation along y-axis
    :param volume: 3D volume to be sliced
    :return: the sliced volume.
    """
    # Euler's transformation
    # Rotation is defined by three rotations around z1, x2, z2 axis
    th_z1 = np.deg2rad(z_rotation)
    th_x2 = np.deg2rad(x_rotation)

    o = np.array(volume.GetOrigin())

    # transformation simplified at z2=0 since this rotation is never performed
    eul_tr = np.array(
        [
            [
                np.cos(th_z1),
                -np.sin(th_z1) * np.cos(th_x2),
                np.sin(th_z1) * np.sin(th_x2),
                o[0] + x_trans,
            ],
            [
                np.sin(th_z1),
                np.cos(th_z1) * np.cos(th_x2),
                -np.cos(th_z1) * np.sin(th_x2),
                o[1] + y_trans,
            ],
            [0, np.sin(th_x2), np.cos(th_x2), o[2]],
            [0, 0, 0, 1],
        ],
    )

    # Define plane's coordinate system
    e1 = eul_tr[0][:3]
    e2 = eul_tr[1][:3]
    e3 = eul_tr[2][:3]
    img_o = eul_tr[:, -1:].flatten()[:3]  # origin of the image plane

    direction = np.stack([e1, e2, e3], axis=0).flatten()

    resampler = sitk.ResampleImageFilter()
    spacing = volume.GetSpacing()

    resampler.SetOutputDirection(direction.tolist())
    resampler.SetOutputOrigin(img_o.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(volume.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Resample the volume on the arbitrary plane
    return resampler.Execute(volume)


def slice_volume(
    volume: sitk.Image,
    slice_shape: tuple[int, int],
    z_rotation: float | np.ndarray = 0.0,
    x_rotation: float | np.ndarray = 0.0,
    x_trans: float | np.ndarray = 0.0,
    y_trans: float | np.ndarray = 0.0,
) -> sitk.Image:
    """Slice a 3D volume with arbitrary rotation and translation.

    :param z_rotation: rotation around z-axis in degrees
    :param x_rotation: rotation around x-axis in degrees
    :param x_trans: translation along x-axis
    :param y_trans: translation along y-axis
    :param volume: 3D volume to be sliced
    :param slice_shape: shape of the output slice
    :return: the sliced volume.
    """
    # Euler's transformation
    # Rotation is defined by three rotations around z1, x2, z2 axis
    th_z1 = np.deg2rad(z_rotation)
    th_x2 = np.deg2rad(x_rotation)

    o = np.array(volume.GetOrigin())

    # transformation simplified at z2=0 since this rotation is never performed
    eul_tr = np.array(
        [
            [
                np.cos(th_z1),
                -np.sin(th_z1) * np.cos(th_x2),
                np.sin(th_z1) * np.sin(th_x2),
                o[0] + x_trans,
            ],
            [
                np.sin(th_z1),
                np.cos(th_z1) * np.cos(th_x2),
                -np.cos(th_z1) * np.sin(th_x2),
                o[1] + y_trans,
            ],
            [0, np.sin(th_x2), np.cos(th_x2), o[2]],
            [0, 0, 0, 1],
        ],
    )

    # Define plane's coordinate system
    e1 = eul_tr[0][:3]
    e2 = eul_tr[1][:3]
    e3 = eul_tr[2][:3]
    img_o = eul_tr[:, -1:].flatten()[:3]  # origin of the image plane

    direction = np.stack([e1, e2, e3], axis=0).flatten()

    resampler = sitk.ResampleImageFilter()
    spacing = volume.GetSpacing()

    w = slice_shape[0]
    h = slice_shape[1]

    resampler.SetOutputDirection(direction.tolist())
    resampler.SetOutputOrigin(img_o.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize((w, 3, h))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Resample the volume on the arbitrary plane
    return resampler.Execute(volume)
