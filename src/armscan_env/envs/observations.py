from collections.abc import Sequence
from functools import cached_property
from typing import (
    Any,
    Generic,
    TypedDict,
    TypeVar,
    cast,
)

import numpy as np
from armscan_env.clustering import TissueClusters, TissueLabel
from armscan_env.envs.base import DictObservation
from armscan_env.envs.rewards import anatomy_based_rwd
from armscan_env.envs.state_action import LabelmapStateAction
from armscan_env.util.img_processing import crop_center

import gymnasium as gym


class ChanneledLabelmapsObsWithActReward(TypedDict):
    """TypeDict for LabelmapSliceAsChannelsObservation.

    * channeled_slice: Boolean 3-d array representing segregated labelmap slice.
    * action: The last action taken.
    * reward: The reward received from the previous action.
    """

    channeled_slice: np.ndarray
    action: np.ndarray
    reward: np.ndarray


class ActionRewardDict(TypedDict):
    action: np.ndarray
    reward: np.ndarray


class ClusteringCharsDict(TypedDict):
    num_clusters: np.ndarray
    num_points: np.ndarray
    cluster_center_mean: np.ndarray


class ClusterObservationDict(ClusteringCharsDict, ActionRewardDict):
    pass


TDict = TypeVar(
    "TDict",
    bound=dict | ChanneledLabelmapsObsWithActReward | ClusteringCharsDict | ActionRewardDict,
)


class MultiBoxSpace(gym.spaces.Dict, Generic[TDict]):
    """The observation space is a dictionary, where each value's space is a Box. The observation space shape is a
    list of the shapes of each Box space. This class is a wrapper around gym.spaces.Dict, with additional methods to
    access the shape of each Box space, which as specified in gym.spaces.Dict needs special handling.

    :param name2box: dictionary of the name of the observation space and the Box space.
    """

    def __init__(self, name2box: dict[str, gym.spaces.Box] | gym.spaces.Dict):
        super().__init__(spaces=name2box)
        self.spaces = name2box

    @property
    def shape(self) -> list[Sequence[int]]:
        """Return the shape of the space as an immutable property.
        This is a special handling.
        """
        return [box.shape for box in self.name2box.values()]

    def get_shape_from_name(self, name: str) -> Sequence[int]:
        """Get the shape of the Box space with the given name."""
        return self.name2box[name].shape

    @property
    def name2box(self) -> dict[str, gym.spaces.Box]:
        """Dictionary of the name of the observation space and the Box space."""
        return self.spaces

    def sample(self, mask: dict[str, Any] | None = None) -> TDict:
        return cast(TDict, super().sample(mask=mask))


class LabelmapSliceAsChannelsObservation(DictObservation[LabelmapStateAction]):
    """Observation space for a labelmap slice, with each tissue type as a separate channel.
    The observation includes the current slice, the last action, and the last reward.
    Each observation is a dictionary of the type ChanneledLabelmapsObsWithActReward.

    :param slice_shape: slices will be cropped to this shape (we need a consistent observation space).
    """

    def __init__(
        self,
        slice_shape: tuple[int, int],
        action_shape: tuple[int],
    ):
        self._observation_space = MultiBoxSpace[ChanneledLabelmapsObsWithActReward](
            {
                "channeled_slice": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(TissueLabel), *slice_shape),
                ),
                "action": gym.spaces.Box(low=-1, high=1, shape=action_shape),
                "reward": gym.spaces.Box(low=-1, high=0, shape=(1,)),
            },
        )

    def compute_observation(
        self,
        state: LabelmapStateAction,
    ) -> ChanneledLabelmapsObsWithActReward:
        """Return the observation as a dictionary of the type ChanneledLabelmapsObsWithActReward."""
        return self.compute_from_slice(
            state.labels_2d_slice,
            state.normalized_action_arr,
        )

    def compute_from_slice(
        self,
        labels_2d_slice: np.ndarray,
        action: np.ndarray,
    ) -> ChanneledLabelmapsObsWithActReward:
        """Compute the observation from the labelmap slice, action, and reward and saves it in a dictionary of the
        form of ChanneledLabelmapsObsWithActReward.
        """
        cropped_slice = crop_center(labels_2d_slice, self.slice_hw)
        channeled_slice = cast(
            np.ndarray[bool, np.dtype[np.bool_]],
            np.zeros(self.channeled_slice_shape, dtype=np.bool_),
        )
        for channel, label in enumerate(TissueLabel):
            channeled_slice[channel] = cropped_slice == label.value

        clusters = TissueClusters.from_labelmap_slice(labels_2d_slice)
        clustering_reward = anatomy_based_rwd(tissue_clusters=clusters)

        return {
            "channeled_slice": channeled_slice,
            "action": np.array(action, dtype=np.float32),
            "reward": np.array([clustering_reward], dtype=np.float32),
        }

    @property
    def slice_hw(self) -> tuple[int, int]:
        return self.channeled_slice_shape[1:]

    @property
    def channeled_slice_shape(self) -> tuple[int, int, int]:
        return cast(
            tuple[int, int, int],
            self.observation_space.get_shape_from_name("channeled_slice"),
        )

    @property
    def action_shape(self) -> tuple[int]:
        return cast(tuple[int], self.observation_space.get_shape_from_name("action"))

    # (channels, h, w), (n_actions), (1)
    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int], tuple[int]]:
        return cast(
            tuple[tuple[int, int, int], tuple[int], tuple[int]],
            self.observation_space.shape,
        )

    @property
    def observation_space(self) -> MultiBoxSpace:
        """Boolean 2-d array representing segregated labelmap slice."""
        return self._observation_space


class LabelmapSliceObservation(DictObservation[LabelmapStateAction]):
    """Observation space for a labelmap slice.
    To test performance diff between this and LabelmapSliceAsChannelsObservation.

    :param slice_shape: slices will be cropped to this shape (we need a consistent observation space).
    """

    def __init__(
        self,
        slice_shape: tuple[int, int],
        action_shape: tuple[int],
    ):
        self._observation_space = MultiBoxSpace[ChanneledLabelmapsObsWithActReward](
            {
                "channeled_slice": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=slice_shape,
                ),
                "action": gym.spaces.Box(low=-1, high=1, shape=action_shape),
                "reward": gym.spaces.Box(low=-1, high=0, shape=(1,)),
            },
        )

    def compute_observation(
        self,
        state: LabelmapStateAction,
    ) -> ChanneledLabelmapsObsWithActReward:
        """Return the observation as a dictionary of the type ChanneledLabelmapsObsWithActReward."""
        return self.compute_from_slice(
            state.labels_2d_slice,
            state.normalized_action_arr,
        )

    def compute_from_slice(
        self,
        labels_2d_slice: np.ndarray,
        action: np.ndarray,
    ) -> ChanneledLabelmapsObsWithActReward:
        """Compute the observation from the labelmap slice, action, and reward and saves it in a dictionary of the
        form of ChanneledLabelmapsObsWithActReward.
        """
        cropped_slice = crop_center(labels_2d_slice, self.slice_hw)

        clusters = TissueClusters.from_labelmap_slice(labels_2d_slice)
        clustering_reward = anatomy_based_rwd(tissue_clusters=clusters)

        return {
            "channeled_slice": np.array(cropped_slice, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": np.array([clustering_reward], dtype=np.float32),
        }

    @property
    def slice_hw(self) -> tuple[int, int]:
        return self.slice_shape

    @property
    def slice_shape(self) -> tuple[int, int]:
        return cast(
            tuple[int, int],
            self.observation_space.get_shape_from_name("channeled_slice"),
        )

    @property
    def action_shape(self) -> tuple[int]:
        return cast(tuple[int], self.observation_space.get_shape_from_name("action"))

    # (h, w), (n_actions), (1)
    @property
    def shape(self) -> tuple[tuple[int, int], tuple[int], tuple[int]]:
        return cast(
            tuple[tuple[int, int], tuple[int], tuple[int]],
            self.observation_space.shape,
        )

    @property
    def observation_space(self) -> MultiBoxSpace:
        """Boolean 2-d array representing segregated labelmap slice."""
        return self._observation_space


class ActionRewardObservation(DictObservation[LabelmapStateAction]):
    """Observation containing (normalized) action and a computed `reward`.

    The reward is not extracted from the environment, but computed from the state.
    Thus, it could in principle be any scalar value, not necessarily a reward.
    """

    def __init__(self, action_shape: tuple[int] = (4,)):
        # TODO: don't add actions here, instead add in a separate ObservationWrapper. Not urgent
        self._action_shape = action_shape

    @property
    def action_shape(self) -> tuple[int]:
        return self._action_shape

    @cached_property
    def observation_space(self) -> MultiBoxSpace:
        return MultiBoxSpace[ActionRewardDict](
            name2box={
                "reward": gym.spaces.Box(low=-1, high=0, shape=(1,)),
                "action": gym.spaces.Box(low=-1, high=1, shape=self.action_shape),
            },
        )

    def compute_observation(self, state: LabelmapStateAction) -> ActionRewardDict:
        tissue_clusters = TissueClusters.from_labelmap_slice(state.labels_2d_slice)
        clustering_reward = anatomy_based_rwd(tissue_clusters=tissue_clusters)
        return {
            "action": state.normalized_action_arr,
            "reward": np.array([clustering_reward], dtype=np.float32),
        }


class LabelmapClusterObservation(DictObservation[LabelmapStateAction]):
    """Observation for a flat array representation of a clustered labelmap slice."""

    @cached_property
    def observation_space(self) -> MultiBoxSpace:
        return MultiBoxSpace[ClusteringCharsDict](
            name2box={
                "num_clusters": gym.spaces.Box(low=0, high=np.inf, shape=(len(TissueLabel),)),
                "num_points": gym.spaces.Box(low=0, high=np.inf, shape=(len(TissueLabel),)),
                "cluster_center_mean": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2 * len(TissueLabel),),
                ),
            },
        )

    def compute_observation(self, state: LabelmapStateAction) -> ClusteringCharsDict:
        """At the moment returns an array of len 3x4.

        The result is: `(num_clusters, num_points in all clusters, cluster_center_mean_x, cluster_center_mean_y)`
        for each tissue type, where values are zero if there are no clusters.
        """
        tissue_clusters = TissueClusters.from_labelmap_slice(state.labels_2d_slice)

        tissues_num_points = np.zeros(len(TissueLabel), dtype=int)
        tissues_num_clusters = np.zeros(len(TissueLabel), dtype=int)
        tissues_cluster_centers = np.zeros((len(TissueLabel), 2), dtype=float)
        for i, tissue_label in enumerate(TissueLabel):
            clusters = tissue_clusters.get_cluster_for_label(tissue_label)
            num_points = 0
            num_clusters = len(clusters)

            if num_clusters > 0:
                cluster_centers = []
                for cluster in clusters:
                    num_points += len(cluster.datapoints)
                    cluster_centers.append(cluster.center)
                clusters_center_mean = np.mean(np.array(cluster_centers), axis=0)
            else:
                clusters_center_mean = np.zeros(2, dtype=float)

            tissues_num_clusters[i] = num_clusters
            tissues_num_points[i] = num_points
            tissues_cluster_centers[i] = clusters_center_mean

        # keep in sync with observation_space
        return {
            "num_clusters": tissues_num_clusters,
            "num_points": tissues_num_points,
            "cluster_center_mean": tissues_cluster_centers.flatten(),
        }
