# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk
from armscan_env.envs.base import Observation, RewardMetric, TerminationCriterion
from armscan_env.envs.labelmaps_navigation import (
    LabelmapClusteringBasedReward,
    LabelmapEnv,
    LabelmapEnvTerminationCriterion,
    LabelmapStateAction,
)
from armscan_env.network import DQN, layer_init
from gymnasium import Env
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
        **make_kwargs: Any,
    ) -> None:
        """:param name2volume: the gymnasium task/environment identifier
        :param render_mode_train: the render mode to use for training environments
        :param render_mode_test: the render mode to use for test environments
        :param render_mode_watch: the render mode to use for environments that are used to watch agent performance
        :param venv_type: the type of vectorized environment to use
        :param make_kwargs: additional keyword arguments to pass to the environment creation function
        """
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
        self.make_kwargs = make_kwargs

    def _create_kwargs(self) -> dict:
        """Adapts the keyword arguments for the given mode.

        :return: adapted keyword arguments
        """
        return dict(self.make_kwargs)

    def create_env(self, mode: EnvMode) -> Env:
        """Creates a single environment for the given mode.

        :param mode: the mode
        :return: an environment
        """
        return LabelmapEnv(
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


class ActorFactoryArmscanDQN(ActorFactory):
    def __init__(
        self,
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
    ) -> None:
        self.output_dim_added_layer = output_dim_added_layer
        self.features_only = features_only

    def create_module(self, envs: Environments, device: TDevice) -> ActorProb:
        c, h, w = envs.get_observation_shape()  # type: ignore  # only right shape is a sequence of length 3
        action_shape = envs.get_action_shape()
        if isinstance(action_shape, np.int64):
            action_shape = int(action_shape)
        net: DQN = DQN(
            c=c,
            h=h,
            w=w,
            action_shape=action_shape,
            device=device,
            features_only=self.features_only,
            output_dim_added_layer=self.output_dim_added_layer,
            layer_init=layer_init,
        )
        return ActorProb(net, envs.get_action_shape(), device=device).to(device)
