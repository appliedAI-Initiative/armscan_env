import logging
from dataclasses import dataclass
from typing import Self

import numpy as np
from armscan_env.envs.base import StateAction

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ManipulatorAction:
    rotation: np.ndarray
    """Array of shape (2,) representing two angles in degrees (z_rot, x_rot). The angles will take values between
    -180 and 180 deg."""
    translation: np.ndarray
    """Array of shape (2,) representing two translations (x_trans, y_trans) in mm."""

    def to_normalized_array(
        self,
        angle_bounds: tuple[float, float],
        translation_bounds: tuple[float | None, float | None],
    ) -> np.ndarray:
        """Converts the action to a 1D array. If angle_bounds is not None, the angles will be normalized to the range
        [-1, 1] using the provided bounds.
        """
        if None in translation_bounds:
            raise ValueError("Translation bounds must not be None,this should not happen.")
        rotation = self.rotation / angle_bounds
        # normalize translation to [-1, 1]: 0 -> -1, translation_bounds -> 1
        translation = 2 * self.translation / translation_bounds - 1  # type: ignore

        result = np.concatenate([rotation, translation])

        if not (result >= -1).all() or not (result <= 1).all():
            raise ValueError(
                f"Angles or translations are out of bounds: "
                f"{self.rotation=}, {self.translation=},"
                f" {angle_bounds=}, {translation_bounds=}",
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
            raise ValueError(
                f"Action is not normalized: {action=}\nShould be in the range [-1, 1]",
            )
        if None in translation_bounds:
            raise ValueError("Translation bounds must not be None,this should not happen.")

        rotation = action[:2] * angle_bounds
        # unnormalize translation: -1 -> 0, 1 -> translation_bounds
        translation = (action[2:] + 1) / 2 * translation_bounds
        log.debug(f"Unnormalized action: {rotation=} deg, {translation=}")

        return cls(rotation=rotation, translation=translation)  # type: ignore


@dataclass(kw_only=True)
class LabelmapStateAction(StateAction):
    action: np.ndarray
    """Array of shape (4,) representing two angles and two translations"""
    labels_2d_slice: np.ndarray
    """Two-dimensional slice of the labelmap, i.e., an array of shape (N, M) with integer values.
    Each integer represents a different label (bone, nerve, etc.)"""
    reward: float | None = None
    """The reward for the current state-action pair. May be None if the reward is not known."""
    optimal_position: np.ndarray | None = None
    """The optimal position for the 2D slice, i.e., the position where the slice is the most informative.
    May be None if the optimal position is not known."""
    optimal_labelmap: np.ndarray | None = None
    """The labelmap at the optimal position. May be None if the optimal position is not known."""
