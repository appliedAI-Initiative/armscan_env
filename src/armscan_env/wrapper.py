# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk
from armscan_env.envs.base import Observation, RewardMetric, TerminationCriterion
from armscan_env.envs.labelmaps_navigation import (
    LabelmapEnv,
    LabelmapEnvTerminationCriterion,
)
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.envs.state_action import LabelmapStateAction
from armscan_env.network import DQN_MLP_Concat
from gymnasium import ActionWrapper, Env, spaces
from gymnasium.wrappers import FrameStack
from tianshou.highlevel.env import (
    EnvFactory,
    Environments,
    EnvMode,
    VectorEnvType,
)
from tianshou.highlevel.module.actor import ActorFactory
from tianshou.highlevel.module.core import TDevice
from tianshou.utils.net.continuous import ActorProb

log = logging.getLogger(__name__)


class ArmscanEnvFactory(EnvFactory):
    """:param name2volume: the gymnasium task/environment identifier
    :param observation: the observation space to use
    :param reward_metric: the reward metric to use
    :param termination_criterion: the termination criterion to use
    :param slice_shape: the shape of the slice
    :param max_episode_len: the maximum episode length
    :param angle_bounds: the bounds for the angles
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
        angle_bounds: tuple[float, float] = (180, 180),
        translation_bounds: tuple[float | None, float | None] = (None, None),
        render_mode_train: Literal["plt", "animation"] | None = None,
        render_mode_test: Literal["plt", "animation"] | None = None,
        render_mode_watch: Literal["plt", "animation"] | None = "animation",
        venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM,
        seed: int | None = None,
        n_stack: int = 1,
        project_to_x_translation: bool = False,
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
        self.angle_bounds = angle_bounds
        self.translation_bounds = translation_bounds
        self.render_modes = {
            EnvMode.TRAIN: render_mode_train,
            EnvMode.TEST: render_mode_test,
            EnvMode.WATCH: render_mode_watch,
        }
        self.seed = seed
        self.n_stack = n_stack
        self.project_to_x_translation = project_to_x_translation
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
            angle_bounds=self.angle_bounds,
            translation_bounds=self.translation_bounds,
            render_mode=self.render_modes.get(mode),
            seed=self.seed,
        )

        if self.project_to_x_translation:
            env = ProjectToXTranslationEnvWrapper(env)

        if self.remove_rotation_actions:
            env = RemoveRotationActionsEnvWrapper(env)

        if self.n_stack > 1:
            env = FrameStack(env, self.n_stack)

        return env


class ProjectToXTranslationEnvWrapper(ActionWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def action(
        self,
        action: spaces.Space[np.ndarray],
    ) -> spaces.Dict:
        """Project the action to the x translation removing the y translation."""
        return spaces.Dict(
            {
                "angle": spaces.Box(
                    low=-1,
                    high=1.0,
                    shape=(2,),
                ),
                "x_translation": spaces.Box(
                    low=-1,
                    high=1.0,
                    shape=(1,),
                ),
            },
        )


class RemoveRotationActionsEnvWrapper(ActionWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def action(
        self,
        action: spaces.Space[np.ndarray],
    ) -> spaces.Dict:
        """Remove the rotation actions."""
        return spaces.Dict(
            {
                "translation": spaces.Box(
                    low=-1,
                    high=1.0,
                    shape=(2,),
                ),
            },
        )


class ActorFactoryArmscanDQN(ActorFactory):
    """A factory for creating DQN_MLP_Concat actors for the armscan_env."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def create_module(self, envs: Environments, device: TDevice) -> ActorProb:
        """Creates a DQN_MLP_Concat actor for the given environments."""
        # happens because the envs will be built based on LabelmapEnv and its observation_space attr
        # which then delivers this kind of tuple of tuples
        # Will fail with any other envs object but we can't currently express this in typing
        # TODO: improve tianshou typing to solve this in env.TObservationShape
        (c, h, w), (action_dim,), _ = envs.get_observation_shape()  # type: ignore
        net = DQN_MLP_Concat(
            c=c,
            h=h,
            w=w,
            action_dim=action_dim,
            device=device,
        )
        return ActorProb(net, envs.get_action_shape(), device=device).to(device)
