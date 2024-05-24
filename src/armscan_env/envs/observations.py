from typing import cast

import gymnasium as gym
import numpy as np
from armscan_env.clustering import TissueClusters, TissueLabel
from armscan_env.envs.base import ArrayObservation, DictObservation, TStateAction
from armscan_env.envs.state_action import LabelmapStateAction
from armscan_env.util.img_processing import crop_center
from numpy import bool_, dtype, ndarray


class LabelmapSliceObservation(ArrayObservation[LabelmapStateAction]):
    def __init__(self, slice_shape: tuple[int, int]):
        """:param slice_shape: slices will be cropped to this shape (we need a consistent observation space)."""
        self._slice_shape = slice_shape
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=slice_shape)

    def compute_observation(self, state: LabelmapStateAction) -> np.ndarray:
        return self.compute_from_slice(state.labels_2d_slice)

    def compute_from_slice(self, labels_2d_slice: np.ndarray) -> np.ndarray:
        return crop_center(labels_2d_slice, self.slice_shape)

    @property
    def slice_shape(self) -> tuple[int, int]:
        return self._slice_shape

    @property
    def observation_space(self) -> gym.spaces.Space[np.ndarray]:
        """Boolean 2-d array representing segregated labelmap slice."""
        return self._observation_space


class LabelmapDictSpace(gym.spaces.Dict):
    def __init__(
        self,
        slice_shape: tuple[int, int],
        action_shape: tuple[int],
        reward_shape: tuple[int],
    ):
        self._slice_shape = slice_shape
        self._channeled_slice_shape = (len(TissueLabel),) + slice_shape  # noqa
        self._action_shape = action_shape
        self._reward_shape = reward_shape
        super().__init__(
            {
                "channeled_slice": gym.spaces.Box(low=0, high=1, shape=self._channeled_slice_shape),
                "action": gym.spaces.Box(low=-1, high=1, shape=self._action_shape),
                "reward": gym.spaces.Box(low=-1, high=0, shape=self._reward_shape),
            },
        )

    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int]]:  # type: ignore
        return self._channeled_slice_shape, self._action_shape, self._reward_shape


class LabelmapSliceAsChannelsObservation(DictObservation[LabelmapStateAction]):
    def __init__(
        self,
        slice_shape: tuple[int, int],
        action_shape: tuple[int],
        reward_shape: tuple[int],
    ):
        """:param slice_shape: slices will be cropped to this shape (we need a consistent observation space)."""
        self._slice_shape = slice_shape
        self._channeled_slice_shape = (len(TissueLabel),) + slice_shape  # noqa
        self._action_shape = action_shape
        self._reward_shape = reward_shape
        self._observation_space = LabelmapDictSpace(
            slice_shape=slice_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
        )

    def compute_observation(
        self,
        state: LabelmapStateAction,
    ) -> tuple[ndarray[bool, dtype[bool_ | bool_]], ndarray, float | None]:
        return self.compute_from_slice(state.labels_2d_slice, state.action, state.reward)

    def compute_from_slice(
        self,
        labels_2d_slice: np.ndarray,
        action: np.ndarray,
        reward: float | None,
    ) -> tuple[ndarray[bool, dtype[bool_]], ndarray, float | None]:
        cropped_slice = crop_center(labels_2d_slice, self.slice_shape)
        channeled_slice = cast(
            np.ndarray[bool, np.dtype[np.bool_]],
            np.zeros(self.channeled_slice_shape, dtype=np.bool_),
        )
        for channel, label in enumerate(TissueLabel):
            channeled_slice[channel] = cropped_slice == label.value
        return channeled_slice, action, reward

    @property
    def slice_shape(self) -> tuple[int, int]:
        return self._slice_shape

    @property
    def channeled_slice_shape(self) -> tuple[int, int, int]:  # Todo: not sure about this
        return self._channeled_slice_shape

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    @property
    def reward_shape(self) -> tuple[int]:
        return self._reward_shape

    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int]]:
        return self.channeled_slice_shape, self.action_shape, self.reward_shape

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Boolean 2-d array representing segregated labelmap slice."""
        return self._observation_space


class LabelmapClusterObservation(ArrayObservation[LabelmapStateAction]):
    def compute_observation(self, state: TStateAction) -> np.ndarray:
        tissue_clusters = TissueClusters.from_labelmap_slice(state.labels_2d_slice)
        return self.cluster_characteristics_array(tissue_cluster=tissue_clusters)

    @staticmethod
    def cluster_characteristics_array(tissue_cluster: TissueClusters) -> np.ndarray:
        characteristics_array = np.zeros((3, 2))
        characteristics_array[0, 0] = len(tissue_cluster.bones)
        characteristics_array[1, 0] = len(tissue_cluster.tendons)
        characteristics_array[2, 0] = len(tissue_cluster.ulnar)
        return characteristics_array
