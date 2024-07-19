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


class ImageVolume(sitk.Image):
    """Represents a 3D volume."""

    def __init__(
        self,
        volume: sitk.Image,
        *args: Any,
        optimal_action: ManipulatorAction | None = None,
    ):
        super().__init__(volume, *args)
        if optimal_action is None:
            optimal_action = ManipulatorAction(rotation=(0.0, 0.0), translation=(0.0, 0.0))
        self._optimal_action = optimal_action

    @property
    def optimal_action(self) -> ManipulatorAction:
        return self._optimal_action

    def get_volume_slice(
        self,
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
            slice_shape = (self.GetSize()[0], self.GetSize()[2])

        origin = np.array(self.GetOrigin())
        rotation = np.deg2rad(action.rotation)
        translation = action.translation

        transform = sitk.Euler3DTransform()
        transform.SetRotation(rotation[1], 0, rotation[0])
        transform.SetTranslation((*translation, 0))
        transform.SetCenter(origin)
        slice_volume = sitk.Resample(
            self,
            transform,
            sitk.sitkNearestNeighbor,
            0.0,
            self.GetPixelID(),
        )

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(slice_volume)
        resampler.SetSize((slice_shape[0], 2, slice_shape[1]))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        # Resample the volume on the arbitrary plane
        return resampler.Execute(slice_volume)[:, 0, :]


class TransformedVolume(ImageVolume):
    """Represents a volume that has been transformed by an action."""

    def __init__(
        self,
        volume: ImageVolume,
        transformation_action: ManipulatorAction | None,
        *args: Any,
    ):
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
        if transformation_action is None:
            transformation_action = ManipulatorAction(rotation=(0.0, 0.0), translation=(0.0, 0.0))

        origin = np.array(volume.GetOrigin())
        rotation = np.deg2rad(transformation_action.rotation)
        translation = transformation_action.translation

        transform = sitk.Euler3DTransform()
        transform.SetRotation(rotation[1], 0, rotation[0])
        transform.SetTranslation((*translation, 0))
        transform.SetCenter(origin)
        resampled = sitk.Resample(
            volume,
            transform,
            sitk.sitkNearestNeighbor,
            0.0,
            volume.GetPixelID(),
        )
        resampled = ImageVolume(resampled, optimal_action=volume.optimal_action)

        super().__init__(resampled, *args)
        self._transformation_action = transformation_action
        self._tr_optimal_action = self.transform_action(volume.optimal_action)

    @property
    def transformation_action(self) -> ManipulatorAction:
        return self._transformation_action

    @property
    def optimal_action(self) -> ManipulatorAction:
        return self._tr_optimal_action

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

        if (
            any(transformed_action.translation < 0)
            or transformed_action.translation[0] > self.GetSize()[0]
            or transformed_action.translation[1] > self.GetSize()[1]
        ):
            log.debug(
                "Action contains a  translation out of volume bounds.\n"
                "Projecting the origin of the viewing plane into the bounds.",
            )
            transformed_action = self.project_into_volume_bounds(transformed_action)

        return transformed_action

    def project_into_volume_bounds(
        self,
        transformed_action: ManipulatorAction,
    ) -> ManipulatorAction:
        """Project the action to the positive octant.
        This is needed when transforming the optimal action accordingly to the random volume transformation.
        It might be, that for a negative translation and/or a negative z-rotation, the coordinates defining the
        optimal action land in negative space. Since the action defines a coordinate frame which infers a plane
        (x-z plane, y normal to the plane), assuming that this plane is still intercepting the positive octant,
        it is possible to redefine the action in positive coordinates by projecting it into the positive octant.

        It needs to be tested, that the volume transformations keep the optimal action in a reachable space.
        Volume transformations are used for data augmentation only, so can be defined in the most convenient way.
        """
        v_size = self.GetSize()
        v_spacing = self.GetSpacing()
        sx, sy = v_size[0] * v_spacing[0], v_size[1] * v_spacing[1]
        tx, ty = transformed_action.translation
        thz, thx = transformed_action.rotation
        log.debug(f"Translation before projection: {transformed_action.translation}")

        while tx < 0 or ty < 0 or tx > sx or ty > sy:
            prev_tx, prev_ty = tx, ty
            if tx < 0:
                ty = (np.tan(np.deg2rad(thz)) * (-tx)) + ty
                tx = 0
            if ty < 0:
                tx = ((1 / np.tan(np.deg2rad(thz))) * (-ty)) + tx
                ty = 0
            if tx > sx:
                ty = (np.tan(np.deg2rad(thz)) * (sx - tx)) + ty
                tx = sx
            if ty > sy:
                tx = ((1 / np.tan(np.deg2rad(thz))) * (sy - ty)) + tx
                ty = sy
            if tx == prev_tx and ty == prev_ty:
                raise ValueError("Loop is stuck, reiterating through the same values.")

        translation = (tx, ty)
        log.debug(f"Translation after projection: {translation}")
        transformed_action.translation = translation
        return transformed_action
