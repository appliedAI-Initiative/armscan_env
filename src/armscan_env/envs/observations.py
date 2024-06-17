from collections.abc import Sequence
from typing import (
    Any,
    Generic,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from armscan_env.clustering import TissueClusters, TissueLabel
from armscan_env.envs.base import ArrayObservation, DictObservation
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


TDict = TypeVar("TDict", bound=Union[dict, ChanneledLabelmapsObsWithActReward])  # noqa


class MultiBoxSpace(gym.spaces.Dict, Generic[TDict]):
    """The observation space is a dictionary, where each value's space is a Box. The observation space shape is a
    list of the shapes of each Box space. This class is a wrapper around gym.spaces.Dict, with additional methods to
    access the shape of each Box space, which as specified in gym.spaces.Dict needs special handling.

    :param name2box: dictionary of the name of the observation space and the Box space.
    """

    def __init__(self, name2box: dict[str, gym.spaces.Box] | gym.spaces.Dict):
        super().__init__(spaces=name2box)  # type: ignore
        self.spaces = name2box  # type: ignore

    @property
    def shape(self) -> list[Sequence[int]]:  # type: ignore # ToDo: improve Tianshou Space.shape
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
        return self.spaces  # type: ignore

    def sample(self, mask: dict[str, Any] | None = None) -> TDict:  # type: ignore # signature expects dict[str, Any]
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
            state.last_reward,
        )

    def compute_from_slice(
        self,
        labels_2d_slice: np.ndarray,
        action: np.ndarray,
        last_reward: float,
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
        return {
            "channeled_slice": channeled_slice,
            "action": np.array(action, dtype=np.float32),
            "reward": np.array([last_reward], dtype=np.float32),
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
            state.last_reward,
        )

    def compute_from_slice(
        self,
        labels_2d_slice: np.ndarray,
        action: np.ndarray,
        last_reward: float,
    ) -> ChanneledLabelmapsObsWithActReward:
        """Compute the observation from the labelmap slice, action, and reward and saves it in a dictionary of the
        form of ChanneledLabelmapsObsWithActReward.
        """
        cropped_slice = crop_center(labels_2d_slice, self.slice_hw)
        return {
            "channeled_slice": np.array(cropped_slice, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": np.array([last_reward], dtype=np.float32),
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


class LabelmapClusterObservation(ArrayObservation[LabelmapStateAction]):
    """Observation for a flat array representation of a clustered labelmap slice.

    The observation contains meaningful characteristics of the slice, as well as the last actions and rewards
    (the past actions are useful if observations are stacked, which is usually the case).
    """

    def __init__(self, action_shape: tuple[int]):
        self.action_shape = action_shape

    def compute_observation(self, state: LabelmapStateAction) -> np.ndarray:
        tissue_clusters = TissueClusters.from_labelmap_slice(state.labels_2d_slice)
        return np.concatenate(
            (
                self.cluster_characteristics_array(tissue_clusters=tissue_clusters).flatten(),
                np.atleast_1d(state.normalized_action_arr),
                np.atleast_1d(state.last_reward),
            ),
            axis=0,
        )

    def _compute_observation_space(
        self,
    ) -> gym.spaces.Box:
        """Return the observation space as a Box, with the right bounds for each feature."""
        DictObs = gym.spaces.Dict(
            spaces=(
                ("num_clusters", gym.spaces.Box(low=0, high=np.inf, shape=(3,))),
                ("num_points", gym.spaces.Box(low=0, high=np.inf, shape=(3,))),
                ("cluster_center_mean", gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))),
                ("action", gym.spaces.Box(low=-1, high=1, shape=self.action_shape)),
                ("reward", gym.spaces.Box(low=-1, high=0, shape=(1,))),
            ),
        )
        return cast(gym.spaces.Box, gym.spaces.flatten_space(DictObs))

    @staticmethod
    def cluster_characteristics_array(tissue_clusters: TissueClusters) -> np.ndarray:
        cluster_characteristics = []

        for tissue_label in TissueLabel:
            clusters = tissue_clusters.get_cluster_for_label(tissue_label)
            num_points = 0
            cluster_centers = []
            for cluster in clusters:
                num_points += len(cluster.datapoints)
                cluster_centers.append(cluster.center)
            clusters_center_mean = np.mean(np.array(cluster_centers), axis=0)
            if np.any(np.isnan(clusters_center_mean)):
                clusters_center_mean = np.zeros(2)
            cluster_characteristics.append([len(clusters), num_points, *clusters_center_mean])

        return np.array(cluster_characteristics)

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Boolean 2-d array representing segregated labelmap slice."""
        return self._compute_observation_space()
