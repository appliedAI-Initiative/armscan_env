# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Any, Final, Literal, SupportsFloat, TypeVar, cast

import numpy as np
import SimpleITK as sitk
from armscan_env.clustering import TissueClusters
from armscan_env.envs.base import (
    ConcatenatedArrayObservation,
    Observation,
    RewardMetric,
    TerminationCriterion,
)
from armscan_env.envs.labelmaps_navigation import (
    LabelmapEnv,
    LabelmapEnvTerminationCriterion,
)
from armscan_env.envs.observations import (
    ActionRewardObservation,
    MultiBoxSpace,
)
from armscan_env.envs.rewards import LabelmapClusteringBasedReward, anatomy_based_rwd
from armscan_env.envs.state_action import LabelmapStateAction

import gymnasium as gym
from gymnasium.core import Env, Wrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from gymnasium.wrappers.utils import create_zero_array
from tianshou.highlevel.env import (
    EnvFactory,
    EnvMode,
    VectorEnvType,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

log = logging.getLogger(__name__)


# Todo: Issue on gymnasium for not overwriting reset method
class PatchedWrapper(Wrapper[np.ndarray, float, np.ndarray, np.ndarray]):
    def __init__(self, env: LabelmapEnv | Env):
        super().__init__(env)
        # Helps with IDE autocompletion
        self.env = cast(LabelmapEnv, env)

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        return self.env.reset(**kwargs)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)


class PatchedActionWrapper(PatchedWrapper, ABC):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(self.action(action))

    @abstractmethod
    def action(self, action: ActType) -> np.ndarray:
        pass


class PatchedFrameStackObservation(PatchedWrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        stack_size: int,
        *,
        padding_type: str | ObsType = "reset",
    ):
        """Had to copy-paste and adjust.

        The inheriting from `RecordConstructorArgs` in original FrameStack is not possible together with
        overridden getattr, which we however need in order to not become crazy.
        So there is no way of fixing this after inheriting from original FrameStackObservation, wherefore
        we copy-paste the whole class and adjust it.

        Adjustments:
            1. reset takes **kwargs
            2. render takes **kwargs
            3. __getattr__ is overridden to pass the attribute to the wrapped environment (like in pre-1.0 wrappers)
            4. No inheritance from RecordConstructorArgs
            5. Observation space is converted to MultiBoxSpace if it is a DictSpace
        """
        super().__init__(env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}",
            )
        if not stack_size > 1:
            raise ValueError(
                f"The stack_size needs to be greater than one, actual value: {stack_size}",
            )
        if isinstance(padding_type, str) and (padding_type == "reset" or padding_type == "zero"):
            self.padding_value: ObsType = create_zero_array(env.observation_space)
        elif padding_type in env.observation_space:
            self.padding_value = padding_type  # type: ignore
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(  # we are guessing that the user just entered the "reset" or "zero" wrong
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}",
                )
            else:  # noqa: RET506
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.observation_space})",
                )

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.stack_size: Final[int] = stack_size
        self.padding_type: Final[str] = padding_type

        self.obs_queue = deque(
            [self.padding_value for _ in range(self.stack_size)],
            maxlen=self.stack_size,
        )
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size)

        if isinstance(self.observation_space, DictSpace):
            self.observation_space = MultiBoxSpace(self.observation_space)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs),
        )
        return updated_obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        if self.padding_type == "reset":
            self.padding_value = obs
        for _ in range(self.stack_size - 1):
            self.obs_queue.append(self.padding_value)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs),
        )
        return updated_obs, info


class PatchedFlattenObservation(PatchedWrapper):
    """Flattens the environment's observation space and each observation from ``reset`` and ``step`` functions.
    Had to copy-paste and adjust.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        PatchedWrapper.__init__(self, env)
        observation_space = gym.spaces.utils.flatten_space(env.observation_space)
        func = lambda obs: gym.spaces.utils.flatten(env.observation_space, obs)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)


class AddObservationsWrapper(Wrapper, ABC):
    """When implementing it, make sure that additional_obs_space is available
    before super().__init__(env) is called.
    """

    def __init__(self, env: LabelmapEnv | Env):
        super().__init__(env)
        if isinstance(self.env.observation_space, Box) and isinstance(
            self.additional_obs_space,
            Box,
        ):
            self.observation_space = ConcatenatedArrayObservation.concatenate_boxes(
                [self.env.observation_space, self.additional_obs_space],
            )
        else:
            raise ValueError(
                f"Observation spaces are not of type Box: {type(self.env.observation_space)}, {type(self.additional_obs_space)}",
            )

    @property
    @abstractmethod
    def additional_obs_space(self) -> gym.spaces:
        pass

    @abstractmethod
    def get_additional_obs_array(
        self,
    ) -> np.ndarray:
        pass

    def observation(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        additional_obs = self.get_additional_obs_array()
        try:
            full_obs = np.concatenate([observation, additional_obs])
        except ValueError:
            raise ValueError(
                f"Observation spaces are not of type Box: {type(observation)}, {type(additional_obs)}",
            ) from None
        return full_obs


class AddRewardDetailsWrapper(AddObservationsWrapper):
    @property
    def additional_obs_space(self) -> gym.spaces:
        return self._additional_obs_space

    def __init__(
        self,
        env: LabelmapEnv,
        num_steps_to_observe: int | None = None,
    ):
        """Adds the action that would lead to the highest image variance to the observation.
        In focus-stigmation agents, this helps in the initial exploratory phase of episodes, as it
        allows wandering around the state space without worrying about losing track of the
        best image found so far.

        :param env:
        :param num_steps_to_observe: Number of steps to observe to pick the highest reward state.
            If None, all steps are observed.
        """
        self.additional_obs = ActionRewardObservation(env.action_space.shape).to_array_observation()
        self._additional_obs_space = self.additional_obs.observation_space
        # don't move above, see comment in AddObservationsWrapper
        super().__init__(env)
        self.num_steps_to_observe = num_steps_to_observe

        self.reset_wrapper()

    def reset_wrapper(self) -> None:
        if self.num_steps_to_observe is None:
            self.rewards: list[float] = []
            self.states: list[LabelmapStateAction] = []
        else:
            self.rewards = deque(maxlen=self.num_steps_to_observe)  # type: ignore
            self.states = deque(maxlen=self.num_steps_to_observe)  # type: ignore

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        self.reset_wrapper()
        obs, info = super().reset(**kwargs)
        updated_obs = cast(ObsType, self.observation(obs))
        return updated_obs, info

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        updated_obs = cast(ObsType, self.observation(obs))
        return updated_obs, reward, terminated, truncated, info

    def _update_observation_fields(self) -> None:
        tissue_clusters = TissueClusters.from_labelmap_slice(
            self.env.cur_state_action.labels_2d_slice,
        )
        clustering_reward = anatomy_based_rwd(tissue_clusters=tissue_clusters)
        self.rewards.append(clustering_reward)
        self.states.append(self.env.cur_state_action)
        self.highest_rew_state_arr = self.states[np.argmax(self.rewards)]

    def get_additional_obs_array(self) -> np.ndarray:
        # base_obs is not used, instead we directly access the current image from the env
        self._update_observation_fields()
        return self.additional_obs.compute_observation(self.highest_rew_state_arr)


class ArmscanEnvFactory(EnvFactory):
    def __init__(
        self,
        name2volume: dict[str, sitk.Image],
        observation: Observation[LabelmapStateAction, Any],
        # TODO: remove mutable default values, make a proper config-based factory (not urgent)
        reward_metric: RewardMetric[LabelmapStateAction] = LabelmapClusteringBasedReward(),
        termination_criterion: TerminationCriterion | None = LabelmapEnvTerminationCriterion(),
        slice_shape: tuple[int, int] | None = None,
        max_episode_len: int | None = None,
        rotation_bounds: tuple[float, float] = (180, 180),
        translation_bounds: tuple[float | None, float | None] = (None, None),
        render_mode_train: Literal["plt", "animation"] | None = None,
        render_mode_test: Literal["plt", "animation"] | None = None,
        render_mode_watch: Literal["plt", "animation"] | None = "animation",
        venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
        seed: int | None = None,
        n_stack: int = 1,
        project_actions_to: Literal["x", "y", "xy", "zy"] | None = None,
        apply_volume_transformation: bool = False,
        add_reward_details: bool = False,
        **make_kwargs: Any,
    ) -> None:
        """:param name2volume: the gymnasium task/environment identifier
        :param observation: the observation to use
        :param reward_metric: the reward metric to use
        :param termination_criterion: the termination criterion to use
        :param slice_shape: the shape of the slice
        :param max_episode_len: the maximum episode length
        :param rotation_bounds: the bounds for the angles
        :param translation_bounds: the bounds for the translations
        :param render_mode_train: the render mode to use for training environments
        :param render_mode_test: the render mode to use for test environments
        :param render_mode_watch: the render mode to use for environments that are used to watch agent performance
        :param venv_type: the type of vectorized environment to use
        :param seed: the seed to use
        :param n_stack: the number of observations to stack in a single observation
        :param project_actions_to: constrains the action space to only x translation
        :param apply_volume_transformation: whether to apply transformations to the volume for data augmentation
        :param add_reward_details: whether to add reward details to the observation
        :param make_kwargs: additional keyword arguments to pass to the environment creation function
        """
        super().__init__(venv_type)
        self.name2volume = name2volume
        self.observation = observation
        self.slice_shape = slice_shape
        self.reward_metric = reward_metric
        self.termination_criterion = termination_criterion
        self.max_episode_len = max_episode_len
        self.rotation_bounds = rotation_bounds
        self.translation_bounds = translation_bounds
        self.render_modes = {
            EnvMode.TRAIN: render_mode_train,
            EnvMode.TEST: render_mode_test,
            EnvMode.WATCH: render_mode_watch,
        }
        self.seed = seed
        self.n_stack = n_stack
        self.project_actions_to = project_actions_to
        self.apply_volume_transformation = apply_volume_transformation
        self.add_reward_details = add_reward_details
        self.make_kwargs = make_kwargs

    def _create_kwargs(self) -> dict:
        """Adapts the keyword arguments for the given mode.

        :return: adapted keyword arguments
        """
        return dict(self.make_kwargs)

    def create_env(self, mode: EnvMode) -> LabelmapEnv:
        """Creates a single environment for the given mode.

        :return: an environment
        """
        env = LabelmapEnv(
            name2volume=self.name2volume,
            observation=self.observation,
            slice_shape=self.slice_shape,
            reward_metric=self.reward_metric,
            termination_criterion=self.termination_criterion,
            max_episode_len=self.max_episode_len,
            rotation_bounds=self.rotation_bounds,
            translation_bounds=self.translation_bounds,
            render_mode=self.render_modes.get(mode),
            seed=self.seed,
            project_actions_to=self.project_actions_to,
            apply_volume_transformation=self.apply_volume_transformation,
        )

        if self.n_stack > 1:
            env = PatchedFrameStackObservation(env, self.n_stack)
        env = PatchedFlattenObservation(env)
        if self.add_reward_details:
            env = AddRewardDetailsWrapper(
                env,
            )
        return env
