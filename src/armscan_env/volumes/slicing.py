import logging
from typing import Any

import numpy as np
import SimpleITK as sitk
from armscan_env.envs.state_action import ManipulatorAction

log = logging.getLogger(__name__)


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
    log.debug(
        f"Original Array Shape: {original_array.shape}\n"
        f"Padded Array Shape: {padded_array.shape}",
    )

    return padded_array


class EulerTransform:
    def __init__(self, action: ManipulatorAction, origin: np.ndarray | None = None):
        if origin is None:
            origin = np.zeros(3)
        self.action = action
        self.origin = origin

    def get_transform_matrix(self) -> np.ndarray:
        # Euler's transformation
        # Rotation is defined by three rotations around z1, x2, z2 axis
        th_z1 = np.deg2rad(self.action.rotation[0])
        th_x2 = np.deg2rad(self.action.rotation[1])

        # transformation simplified at z2=0 since this rotation is never performed
        return np.array(
            [
                [
                    np.cos(th_z1),
                    -np.sin(th_z1) * np.cos(th_x2),
                    np.sin(th_z1) * np.sin(th_x2),
                    self.origin[0] + self.action.translation[0],
                ],
                [
                    np.sin(th_z1),
                    np.cos(th_z1) * np.cos(th_x2),
                    -np.cos(th_z1) * np.sin(th_x2),
                    self.origin[1] + self.action.translation[1],
                ],
                [0, np.sin(th_x2), np.cos(th_x2), self.origin[2]],
                [0, 0, 0, 1],
            ],
        )

    @staticmethod
    def get_angles_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
        """Get the angles from a rotation matrix."""
        # Extract the angles from the rotation matrix
        th_x2 = np.arcsin(rotation_matrix[2, 1])
        th_z1 = np.arcsin(rotation_matrix[1, 0])

        # Convert the angles to degrees
        th_z1 = np.rad2deg(th_z1)
        th_x2 = np.rad2deg(th_x2)

        return np.array([th_z1, th_x2])

    def transform_action(self, relative_action: ManipulatorAction) -> ManipulatorAction:
        """Transform an action to be relative to the new coordinate system."""
        volume_transform_matrix = self.get_transform_matrix()

        action_matrix = EulerTransform(relative_action).get_transform_matrix()
        new_action_matrix = np.dot(
            np.linalg.inv(volume_transform_matrix),
            action_matrix,
        )  # 1_A_s = 1_T_0 * 0_A_s

        # new_action_translation = action_matrix[:2, 3] - volume_transform_matrix[:2, 3]
        new_action_rotation = self.get_angles_from_rotation_matrix(new_action_matrix[:3, :3])
        new_action_translation = new_action_matrix[:2, 3]

        transformed_action = ManipulatorAction(
            rotation=new_action_rotation,
            translation=new_action_translation,
        )

        log.debug(
            f"Random transformation: {self.action}\n"
            f"Original action: {relative_action}\n"
            f"Transformed action: {transformed_action}\n",
        )

        return transformed_action


class TransformedVolume(sitk.Image):
    """Represents a volume that has been transformed by an action.

    Should only ever be instantiated by `create_transformed_volume`.
    """

    def __init__(self, *args: Any, transformation_action: ManipulatorAction, _private: int):
        if _private != 42:
            raise ValueError(
                "TransformedVolume should only be instantiated by create_transformed_volume.",
            )
        super().__init__(*args)
        self._transformation_action = transformation_action

    @property
    def transformation_action(self) -> ManipulatorAction | None:
        return self._transformation_action


def create_transformed_volume(
    volume: sitk.Image,
    transformation_action: ManipulatorAction,
) -> TransformedVolume:
    """Transform a 3D volume with arbitrary rotation and translation.

    :param volume: 3D volume to be transformed
    :param transformation_action: action to transform the volume
    :return: the sliced volume.
    """
    if isinstance(volume, TransformedVolume):
        raise ValueError(
            f"This operation should only be performed on a non-transformed volume "
            f"but got an instance of: {volume.__class__.__name__}.",
        )

    origin = np.array(volume.GetOrigin())
    transform_matrix = EulerTransform(transformation_action, origin).get_transform_matrix()

    # Define plane's coordinate system
    e1 = transform_matrix[0][:3]
    e2 = transform_matrix[1][:3]
    e3 = transform_matrix[2][:3]
    img_o = transform_matrix[:, -1:].flatten()[:3]  # origin of the image plane

    direction = np.stack([e1, e2, e3], axis=0).flatten()

    resampler = sitk.ResampleImageFilter()
    spacing = volume.GetSpacing()

    resampler.SetOutputDirection(direction.tolist())
    resampler.SetOutputOrigin(img_o.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(volume.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Resample the volume on the arbitrary plane
    transformed_volume = resampler.Execute(volume)
    # needed to deal with rotation dependency of the volume
    return TransformedVolume(
        transformed_volume,
        transformation_action=transformation_action,
        _private=42,
    )


def get_volume_slice(
    volume: sitk.Image,
    # TODO: shouldn't there be a default shape, like the native shape of the volume itself?
    slice_shape: tuple[int, int],
    action: ManipulatorAction,
) -> sitk.Image:
    """Slice a 3D volume with arbitrary rotation and translation.

    :param volume: 3D volume to be sliced
    :param slice_shape: shape of the output slice
    :param action: action to transform the volume
    :return: the sliced volume.
    """
    o = np.array(volume.GetOrigin())

    if isinstance(volume, TransformedVolume):
        # TODO: why is origin not used here?
        volume_transformation = EulerTransform(volume.transformation_action).get_transform_matrix()
        action_transformation = EulerTransform(action).get_transform_matrix()
        # TODO: why not use action_transformation.transform_action ?
        inverse_transformed_action = np.dot(volume_transformation, action_transformation)
        action_rotation = EulerTransform.get_angles_from_rotation_matrix(
            inverse_transformed_action[:3, :3],
        )
        action_translation = (inverse_transformed_action[:3, 3] - volume_transformation[:3, 3])[:2]
        action = ManipulatorAction(rotation=action_rotation, translation=(action_translation))

    # TODO: seems to be recomputed as action_transformation in block above - can it be computed once instead?
    euler_transform = EulerTransform(action, o)
    eul_tr = euler_transform.get_transform_matrix()

    # Define plane's coordinate system
    rotation = eul_tr[:3, :3]
    translation = eul_tr[:, -1:].flatten()[:3]  # origin of the image plane

    rotation = rotation.flatten()

    resampler = sitk.ResampleImageFilter()
    spacing = volume.GetSpacing()

    w = slice_shape[0]
    h = slice_shape[1]

    resampler.SetOutputDirection(rotation.tolist())
    resampler.SetOutputOrigin(translation.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize((w, 3, h))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Resample the volume on the arbitrary plane
    return resampler.Execute(volume)
