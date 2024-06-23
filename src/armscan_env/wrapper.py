# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Any, Final, Literal, SupportsFloat, TypeVar, cast

import numpy as np
import SimpleITK as sitk
from armscan_env.envs.base import Observation, RewardMetric, TerminationCriterion
from armscan_env.envs.labelmaps_navigation import (
    LabelmapEnv,
    LabelmapEnvTerminationCriterion,
)
from armscan_env.envs.observations import MultiBoxSpace
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.envs.state_action import LabelmapStateAction

import gymnasium as gym
from gymnasium.core import Env, Wrapper
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


class PatchedFrameStackObservation(Wrapper):
    def __init__(
        self,
        env: gym.core.Env[ObsType, ActType],
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

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    # Like in PatchedWrapper
    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)


class ArmscanEnvFactory(EnvFactory):
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
    :param project_to_x_translation: constrains the action space to only x translation
    :param remove_rotation_actions: removes the rotation actions from the action space
    :param make_kwargs: additional keyword arguments to pass to the environment creation function
    """

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
        project_actions_to: Literal["x", "y", "xy"] | None = None,
        apply_volume_transformation: bool = False,
        **make_kwargs: Any,
    ) -> None:
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
        return env


# Todo: Issue on gymnasium for not overwriting reset method
class PatchedWrapper(Wrapper[np.ndarray, float, np.ndarray, np.ndarray]):
    def __init__(self, env: LabelmapEnv | Env):
        super().__init__(env)
        # Helps with IDE autocompletion
        self.env = cast(LabelmapEnv, env)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(**kwargs)

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
