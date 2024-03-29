import logging
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import gymnasium as gym
import numpy as np
import SimpleITK as sitk
from armscan_env.clustering import DBSCAN_cluster_iter
from armscan_env.envs.base import (
    ArrayObservation,
    ModularEnv,
    RewardMetric,
    StateAction,
    TerminationCriterion,
)
from armscan_env.envs.rewards import anatomy_based_rwd
from armscan_env.slicing import slice_volume
from armscan_env.util.img_processing import crop_center
from gymnasium.core import ObsType as TObs

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ManipulatorAction:
    rotation: np.ndarray
    """Array of shape (2,) representing two angles in degrees. The angles will take values between -180 and 180 deg."""
    translation: np.ndarray
    """Array of shape (2,) representing two translations TODO: extend description"""

    def to_normalized_array(
        self,
        angle_bounds: tuple[float, float],
        translation_bounds: tuple[float, float],
    ) -> np.ndarray:
        """Converts the action to a 1D array. If angle_bounds is not None, the angles will be normalized to the range
        [-1, 1] using the provided bounds.
        """
        rotation = self.rotation / angle_bounds
        translation = self.translation / translation_bounds

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
        angle_bounds: tuple[float, float] | None = None,
        translation_bounds: tuple[float, float] | None = None,
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

        rotation = action[:2] * angle_bounds
        translation = action[2:] * translation_bounds
        log.debug(f"Unnormalized action: {rotation=} deg, {translation=}")

        return cls(rotation=rotation, translation=translation)


@dataclass(kw_only=True)
class LabelmapStateAction(StateAction):
    action: np.ndarray
    """Array of shape (5,) representing two angles and two translations"""
    labels_2d_slice: np.ndarray
    """Two-dimensional slice of the labelmap, i.e., an array of shape (N, M) with integer values.
    Each integer represents a different label (bone, nerve, etc.)"""
    optimal_position: np.ndarray | None = None
    """The optimal position for the 2D slice, i.e., the position where the slice is the most informative.
    May be None if the optimal position is not known."""
    optimal_labelmap: np.ndarray | None = None
    """The labelmap at the optimal position. May be None if the optimal position is not known."""


class LabelmapSliceObservation(ArrayObservation[LabelmapStateAction]):
    def __init__(self, slice_shape: tuple[int, int], render_mode: str | None = None):
        """:param slice_shape: slices will be cropped to this shape (we need a consistent observation space)."""
        self._slice_shape = slice_shape
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=slice_shape)

    @property
    def slice_shape(self) -> tuple[int, int]:
        return self._slice_shape

    def compute_observation(self, state: LabelmapStateAction) -> np.ndarray:
        return crop_center(state.labels_2d_slice, self.slice_shape)

    @property
    def observation_space(self) -> gym.spaces.Space[np.ndarray]:
        """Boolean 2-d array representing segregated labelmap slice."""
        return self._observation_space


class LabelmapClusteringBasedReward(RewardMetric[LabelmapStateAction]):
    def __init__(
        self,
        tissues: dict[str, int],
        n_landmarks: Sequence[int] = (5, 2, 1),
    ):
        self.tissues = tissues
        self.n_landmarks = n_landmarks

    def compute_reward(self, state: LabelmapStateAction) -> float:
        clusters = DBSCAN_cluster_iter(self.tissues, state.labels_2d_slice)
        return anatomy_based_rwd(tissue_clusters=clusters, n_landmarks=self.n_landmarks)

    @property
    def range(self) -> tuple[float, float]:
        return 0.0, 1.0


class LabelmapEnvTerminationCriterion(TerminationCriterion["LabelmapEnv"], ABC):
    pass


class LabelmapEnv(ModularEnv[LabelmapStateAction, np.ndarray, np.ndarray]):
    _INITIAL_POS_ROTATION = np.zeros(4)

    def __init__(
        self,
        name2volume: dict[str, sitk.Image],
        slice_shape: tuple[int, int],
        reward_metric: RewardMetric[LabelmapStateAction],
        termination_criterion: TerminationCriterion | None = None,
        max_episode_len: int | None = None,
        angle_bounds: tuple[float, float] = (180, 180),
        translation_bounds: tuple[float, float] = (10, 10),
        render_mode: str | None = None,
    ):
        """:param name2volume: mapping from labelmap names to volumes. One of these volumes will be selected at reset.
        :param slice_shape: determines the shape of the 2D slices that will be used as observations
        :param reward_metric: defines the reward metric that will be used, e.g. LabelmapClusteringBasedReward
        :param termination_criterion: if None, no termination criterion will be used
        :param max_episode_len:
        """
        if not name2volume:
            raise ValueError("name2volume must not be empty")
        observation = LabelmapSliceObservation(slice_shape, render_mode)
        super().__init__(reward_metric, observation, termination_criterion, max_episode_len)
        self.name2volume = name2volume
        self._slice_shape = slice_shape

        # set at reset
        self._cur_labelmap_name: str | None = None
        self._cur_labelmap_volume: sitk.Image | None = None

        self.angle_bounds = angle_bounds
        self.translation_bounds = translation_bounds

    def unnormalize_rotation_translation(self, action: np.ndarray) -> ManipulatorAction:
        """Unnormalizes an array with values in the range [-1, 1] to the original range that is
        consumed by :func:`slice_volume`.

        :param action: normalized action with values in the range [-1, 1]
        :return: unnormalized action that can be used with :func:`slice_volume`
        """
        return ManipulatorAction.from_normalized_array(
            action,
            self.angle_bounds,
            self.translation_bounds,
        )

    @property
    def cur_labelmap_name(self) -> str | None:
        return self._cur_labelmap_name

    @property
    def cur_labelmap_volume(self) -> sitk.Image | None:
        return self._cur_labelmap_volume

    @property
    def action_space(self) -> gym.spaces.Space[np.ndarray]:
        return gym.spaces.Box(
            low=-1,
            high=1.0,
            shape=(4,),
        )  # 2 rotations, 2 translations. Should be normalized

    def close(self) -> None:
        super().close()
        self.name2volume = {}
        self._cur_labelmap_name = None
        self._cur_labelmap_volume = None

    def _get_slice_from_action(self, action: np.ndarray) -> np.ndarray:
        manipulator_action = self.unnormalize_rotation_translation(action)
        sliced_volume = slice_volume(
            volume=self.cur_labelmap_volume,
            z_rotation=manipulator_action.rotation[0],
            x_rotation=manipulator_action.rotation[1],
            x_trans=manipulator_action.translation[0],
            y_trans=manipulator_action.translation[1],
        )
        return sitk.GetArrayFromImage(sliced_volume)[:, 0, :]

    def _get_initial_slice(self) -> np.ndarray:
        return self._get_slice_from_action(self._INITIAL_POS_ROTATION)

    def compute_next_state(self, action: np.ndarray) -> LabelmapStateAction:
        new_slice = self._get_slice_from_action(action)
        return LabelmapStateAction(
            action=action,
            labels_2d_slice=new_slice,
            optimal_position=self.cur_state_action.optimal_position,
            optimal_labelmap=self.cur_state_action.optimal_labelmap,
        )

    def sample_initial_state(self) -> LabelmapStateAction:
        if not self.name2volume:
            raise RuntimeError(
                "The name2volume attribute must not be empty. Did you close the environment?",
            )
        sampled_image_name = np.random.choice(list(self.name2volume.keys()))
        self._cur_labelmap_name = sampled_image_name
        self._cur_labelmap_volume = self.name2volume[sampled_image_name]
        # Alternatively, select a random slice
        initial_slice = self._get_initial_slice()
        return LabelmapStateAction(
            action=self._INITIAL_POS_ROTATION,
            labels_2d_slice=initial_slice,
            optimal_position=None,
            optimal_labelmap=None,
        )

    def step(
        self,
        action: np.ndarray | ManipulatorAction,
    ) -> tuple[TObs, float, bool, bool, dict[str, Any]]:
        if isinstance(action, ManipulatorAction):
            action = action.to_normalized_array(self.angle_bounds, self.translation_bounds)
        return super().step(action)
