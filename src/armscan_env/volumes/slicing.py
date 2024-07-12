import logging
from typing import Any

import numpy as np
import SimpleITK as sitk
from armscan_env.envs.state_action import ManipulatorAction

log = logging.getLogger(__name__)


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


class TransformedVolume(sitk.Image):
    """Represents a volume that has been transformed by an action.

    Should only ever be instantiated by `create_transformed_volume`.
    """

    def __init__(self, *args: Any, transformation_action: ManipulatorAction | None, _private: int):
        if _private != 42:
            raise ValueError(
                "TransformedVolume should only be instantiated by create_transformed_volume.",
            )
        if transformation_action is None:
            transformation_action = ManipulatorAction(rotation=(0.0, 0.0), translation=(0.0, 0.0))
        super().__init__(*args)
        self._transformation_action = transformation_action

    @property
    def transformation_action(self) -> ManipulatorAction:
        return self._transformation_action

    def transform_action(self, relative_action: ManipulatorAction) -> ManipulatorAction:
        """Transform an action by the inverse of the volume transformation to be relative to the new coordinate
        system.
        """
        origin = np.array(self.GetOrigin())

        volume_rotation = np.deg2rad(self.transformation_action.rotation)
        volume_translation = self.transformation_action.translation
        volume_transform = sitk.Euler3DTransform(
            origin,
            volume_rotation[1],
            0,
            volume_rotation[0],
            (*volume_translation, 0),
        )

        inverse_volume_transform = volume_transform.GetInverse()
        inverse_volume_transform_matrix = np.eye(4)
        inverse_volume_transform_matrix[:3, :3] = np.array(
            inverse_volume_transform.GetMatrix(),
        ).reshape(3, 3)
        inverse_volume_transform_matrix[:3, 3] = inverse_volume_transform.GetTranslation()

        action_rotation = np.deg2rad(relative_action.rotation)
        action_translation = relative_action.translation
        action_transform = sitk.Euler3DTransform(
            origin,
            action_rotation[1],
            0,
            action_rotation[0],
            (*action_translation, 0),
        )

        action_transform_matrix = np.eye(4)
        action_transform_matrix[:3, :3] = np.array(action_transform.GetMatrix()).reshape(3, 3)
        action_transform_matrix[:3, 3] = action_transform.GetTranslation()

        # 1_A_s = 1_T_0 * 0_A_s
        new_action_matrix = np.dot(inverse_volume_transform_matrix, action_transform_matrix)
        transformed_action = ManipulatorAction(
            rotation=EulerTransform.get_angles_from_rotation_matrix(new_action_matrix[:3, :3]),
            translation=new_action_matrix[:2, 3],
        )

        log.debug(
            f"Random transformation: {self.transformation_action}\n"
            f"Original action: {relative_action}\n"
            f"Transformed action: {transformed_action}\n",
        )

        return transformed_action


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
    rotation = np.deg2rad(transformation_action.rotation)
    translation = transformation_action.translation

    transform = sitk.Euler3DTransform()
    transform.SetRotation(rotation[1], 0, rotation[0])
    transform.SetTranslation((*translation, 0))
    transform.SetCenter(origin)
    resampled = sitk.Resample(volume, transform, sitk.sitkNearestNeighbor, 0.0, volume.GetPixelID())
    # needed to deal with rotation dependency of the volume
    return TransformedVolume(
        resampled,
        transformation_action=transformation_action,
        _private=42,
    )


def get_volume_slice(
    volume: sitk.Image,
    action: ManipulatorAction,
    slice_shape: tuple[int, int] | None = None,
) -> sitk.Image:
    """Slice a 3D volume with arbitrary rotation and translation.

    :param volume: 3D volume to be sliced
    :param action: action to transform the volume
    :param slice_shape: shape of the output slice
    :return: the sliced volume.
    """
    if slice_shape is None:
        slice_shape = (volume.GetSize()[0], volume.GetSize()[2])

    origin = np.array(volume.GetOrigin())
    rotation = np.deg2rad(action.rotation)
    translation = action.translation

    transform = sitk.Euler3DTransform()
    transform.SetRotation(rotation[1], 0, rotation[0])
    transform.SetTranslation((*translation, 0))
    transform.SetCenter(origin)
    resampled = sitk.Resample(volume, transform, sitk.sitkNearestNeighbor, 0.0, volume.GetPixelID())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(resampled)
    resampler.SetSize((slice_shape[0], 2, slice_shape[1]))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Resample the volume on the arbitrary plane
    return resampler.Execute(resampled)[:, 0, :]
