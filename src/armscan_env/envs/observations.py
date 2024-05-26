from collections import OrderedDict
from collections.abc import Sequence
from typing import (
    Any,
    Generic,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import gymnasium as gym
import numpy as np
from armscan_env.clustering import TissueClusters, TissueLabel
from armscan_env.envs.base import ArrayObservation, DictObservation, TStateAction
from armscan_env.envs.state_action import LabelmapStateAction
from armscan_env.util.img_processing import crop_center


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

    def __init__(self, name2box: dict[str, gym.spaces.Box]):
        # If we don't do this, gymnasium will order alphabetically in init
        name2box = OrderedDict(**name2box)
        super().__init__(spaces=name2box)  # type: ignore

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
        return self.compute_from_slice(state.labels_2d_slice, state.action, state.last_reward)

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
            "action": action,
            "reward": np.array(last_reward),
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


class LabelmapSliceObservation(ArrayObservation[LabelmapStateAction]):
    """Observation space for a labelmap slice.
    To test performance diff between this and LabelmapSliceAsChannelsObservation.

    :param slice_shape: slices will be cropped to this shape (we need a consistent observation space).
    """

    def __init__(self, slice_shape: tuple[int, int]):
        self._slice_shape = slice_shape
        self._observation_space = gym.spaces.Box(low=0, high=4, shape=slice_shape)

    def compute_observation(self, state: LabelmapStateAction) -> np.ndarray:
        return self.compute_from_slice(state.labels_2d_slice)

    def compute_from_slice(self, labels_2d_slice: np.ndarray) -> np.ndarray:
        return crop_center(labels_2d_slice, self.slice_shape)

    @property
    def slice_shape(self) -> tuple[int, int]:
        return self._slice_shape

    @property
    def observation_space(self) -> gym.spaces.Space[np.ndarray]:
        return self._observation_space


class LabelmapClusterObservation(ArrayObservation[LabelmapStateAction]):
    """Observation space for an array representation of a clustered labelmap slice.
    The array should contain the meaningful characteristics of the slice.
    TODO: Implement this observation.
    """

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
