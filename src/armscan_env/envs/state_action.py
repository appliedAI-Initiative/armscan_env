import logging
import warnings
from dataclasses import dataclass
from typing import Self

import numpy as np
from armscan_env.envs.base import StateAction

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ManipulatorAction:
    rotation: tuple[float, float] | np.ndarray
    """Array of shape (2,) representing two angles in degrees (z_rot, x_rot). The angles will take values between
    -180 and 180 deg."""
    translation: tuple[float, float] | np.ndarray
    """Array of shape (2,) representing two translations (x_trans, y_trans) in mm."""

    def to_normalized_array(
        self,
        rotation_bounds: tuple[float, float],
        translation_bounds: tuple[float | None, float | None],
    ) -> np.ndarray:
        """Converts the action to a 1D array. If angle_bounds is not None, the angles will be normalized to the range
        [-1, 1] using the provided bounds.
        """
        if None in translation_bounds:
            raise ValueError("Translation bounds must not be None,this should not happen.")
        # normalize translation to [-1, 1]: 0 -> -1, translation_bounds -> 1
        rotation = np.zeros(2)
        translation = np.zeros(2)

        for i in range(2):
            if rotation_bounds[i] == 0.0:
                rotation[i] = 0.0
            else:
                rotation[i] = self.rotation[i] / rotation_bounds[i]

            if translation_bounds[i] == 0.0:
                translation[i] = 0.0
            else:
                translation[i] = 2 * self.translation[i] / translation_bounds[i] - 1  # type: ignore

        result = np.concatenate([rotation, translation])

        if not (result >= -1).all() or not (result <= 1).all():
            warnings.warn(
                f"Angles or translations are out of bounds: "
                f"{self.rotation=}, {self.translation=},"
                f" {rotation_bounds=}, {translation_bounds=}",
            )
        return result

    @classmethod
    def from_normalized_array(
        cls,
        action: np.ndarray,
        angle_bounds: tuple[float, float],
        translation_bounds: tuple[float | None, float | None],
    ) -> Self:
        """Converts a 1D array to a ManipulatorAction. If angle_bounds is not None, the angles will be unnormalized
        using the provided bounds.
        """
        if not (action.shape == (4,)):
            raise ValueError(f"Action has wrong shape: {action.shape=}\nShould be (4,)")
        if not (action >= -1).all() or not (action <= 1).all():
            warnings.warn(
                f"Action is not normalized: {action=}\nShould be in the range [-1, 1]",
            )
        if None in translation_bounds:
            raise ValueError("Translation bounds must not be None,this should not happen.")

        rotation = action[:2] * angle_bounds
        # unnormalize translation: -1 -> 0, 1 -> translation_bounds
        translation = (action[2:] + 1) / 2 * translation_bounds
        log.debug(f"Unnormalized action: {rotation=} deg, {translation=}")

        return cls(rotation=tuple(rotation), translation=tuple(translation))  # type: ignore

    @classmethod
    def sample(
        cls,
        rotation_range: tuple[float, float] = (10.0, 0.0),
        translation_range: tuple[float, float] = (10.0, 10.0),
    ) -> Self:
        rotation = (
            np.random.uniform(-rotation_range[0], rotation_range[0]),
            np.random.uniform(-rotation_range[1], rotation_range[1]),
        )
        translation = (
            np.random.uniform(-translation_range[0], translation_range[0]),
            np.random.uniform(-translation_range[1], translation_range[1]),
        )
        return cls(rotation=rotation, translation=translation)


@dataclass(kw_only=True)
class LabelmapStateAction(StateAction):
    normalized_action_arr: np.ndarray
    """Flat normalized array representing a subset (projection) of two angles and two translations."""
    labels_2d_slice: np.ndarray
    """Two-dimensional slice of the labelmap, i.e., an array of shape (N, M) with integer values.
    Each integer represents a different label (bone, nerve, etc.)"""
    optimal_position: np.ndarray | ManipulatorAction | None = None
    """The optimal position for the 2D slice, i.e., the position where the slice is the most informative.
    May be None if the optimal position is not known."""
    optimal_labelmap: np.ndarray | None = None
    """The labelmap at the optimal position. May be None if the optimal position is not known."""
