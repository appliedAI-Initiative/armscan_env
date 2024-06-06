# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import SimpleITK as sitk
from armscan_env.envs.base import Observation, RewardMetric, TerminationCriterion
from armscan_env.envs.labelmaps_navigation import (
    LabelmapEnv,
    LabelmapEnvTerminationCriterion,
)
from armscan_env.envs.observations import MultiBoxSpace
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.envs.state_action import LabelmapStateAction, ManipulatorAction
from gymnasium import ActionWrapper, Env
from gymnasium.wrappers import FrameStackObservation

from tianshou.highlevel.env import (
    EnvFactory,
    EnvMode,
    VectorEnvType,
)

log = logging.getLogger(__name__)


class ArmscanEnvFactory(EnvFactory):
    """:param name2volume: the gymnasium task/environment identifier
    :param observation: the observation space to use
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
        project_to_x_translation: bool = False,
        optimal_action: ManipulatorAction | None = None,
        remove_rotation_actions: bool = False,
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
        self.project_to_x_translation = project_to_x_translation
        self.optimal_action = optimal_action
        self.remove_rotation_actions = remove_rotation_actions
        self.make_kwargs = make_kwargs

    def _create_kwargs(self) -> dict:
        """Adapts the keyword arguments for the given mode.

        :return: adapted keyword arguments
        """
        return dict(self.make_kwargs)

    def create_env(self, mode: EnvMode) -> Env:
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
        )

        if self.project_to_x_translation:
            if self.optimal_action is None:
                raise ValueError(
                    "optimal_action must be provided when project_to_x_translation is True",
                )
            env = LinearSweepWrapper(env, self.optimal_action)

        if self.n_stack > 1:
            env = FrameStackObservation(env, self.n_stack)
            env.observation_space = MultiBoxSpace(env.observation_space)

        return env


class LinearSweepWrapper(ActionWrapper):
    def __init__(self, env: LabelmapEnv, optimal_action: ManipulatorAction) -> None:
        super().__init__(env)
        self.env: LabelmapEnv = env
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        self.optimal_action = optimal_action

    def action(self, action: np.ndarray) -> np.ndarray:
        normalized_optimal_action = self.optimal_action.to_normalized_array(
            self.env.rotation_bounds,
            self.env.translation_bounds,
        )
        return np.append(normalized_optimal_action[:3], action[0])
