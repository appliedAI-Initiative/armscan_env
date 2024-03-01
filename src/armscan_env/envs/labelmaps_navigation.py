from abc import ABC
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import SimpleITK as sitk
from armscan_env.envs.base import (
    ArrayObservation,
    ModularEnv,
    RewardMetric,
    StateAction,
    TerminationCriterion,
    TStateAction,
)
from armscan_env.slicing import slice_volume
from armscan_env.util.img_processing import crop_center


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
    def __init__(self, slice_shape: tuple[int, int]):
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
    def compute_reward(self, state: TStateAction) -> float:
        # TODO: implement with DBSCAN or similar
        return 0.0

    @property
    def range(self) -> tuple[float, float]:
        return 0.0, 1.0


class LabelmapEnvTerminationCriterion(TerminationCriterion["LabelmapEnv"], ABC):
    pass


def unnormalize_rotation_translation(action: np.ndarray) -> np.ndarray:
    """Unnormalizes an array with values in the range [-1, 1] to the original range that is
    consumed by :func:`slice_volume`.

    :param action: normalized action with values in the range [-1, 1]
    :return: unnormalized action that can be used with :func:`slice_volume`
    """
    # TODO: implement
    return action


class LabelmapEnv(ModularEnv[LabelmapStateAction, np.ndarray, np.ndarray]):
    _INITIAL_POS_ROTATION = np.zeros(4)

    def __init__(
        self,
        name2volume: dict[str, sitk.Image],
        slice_shape: tuple[int, int],
        reward_metric: RewardMetric[LabelmapStateAction] | None = None,
        termination_criterion: TerminationCriterion | None = None,
        max_episode_len: int | None = None,
    ):
        """:param name2volume: mapping from labelmap names to volumes. One of these volumes will be selected at reset.
        :param slice_shape: determines the shape of the 2D slices that will be used as observations
        :param reward_metric: if None, a default reward metric will be used
        :param termination_criterion: if None, no termination criterion will be used
        :param max_episode_len:
        """
        if not name2volume:
            raise ValueError("name2volume must not be empty")
        reward_metric = reward_metric or LabelmapClusteringBasedReward()
        observation = LabelmapSliceObservation(slice_shape)
        super().__init__(reward_metric, observation, termination_criterion, max_episode_len)
        self.name2volume = name2volume
        self._slice_shape = slice_shape

        # set at reset
        self._cur_labelmap_name: str | None = None
        self._cur_labelmap_volume: sitk.Image | None = None

    @property
    def cur_labelmap_name(self) -> str | None:
        return self._cur_labelmap_name

    @property
    def cur_labelmap_volume(self) -> sitk.Image | None:
        return self._cur_labelmap_volume

    @property
    def action_space(self) -> gym.spaces.Space[np.ndarray]:
        # TODO: correct bounds
        return gym.spaces.Box(
            low=-1,
            high=1.0,
            shape=(5,),
        )  # 2 rotations, 3 translations. Should be normalized

    def close(self) -> None:
        super().close()
        self.name2volume = {}
        self._cur_labelmap_name = None
        self._cur_labelmap_volume = None

    def _get_slice_from_action(self, action: np.ndarray) -> np.ndarray:
        # TODO: I'm not sure this is correct
        unnormalize_rotation_translation(action)
        sliced_volume = slice_volume(*action, volume=self.cur_labelmap_volume)
        sliced_img = sitk.GetArrayFromImage(sliced_volume)
        # TODO: sliced_img is 3D - why is that? Why do we take the zeroth channel?
        return sliced_img[:, 0, :]

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
