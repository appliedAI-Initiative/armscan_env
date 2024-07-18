import logging
import os

from armscan_env.config import get_config
from armscan_env.envs.labelmaps_navigation import LabelmapEnvTerminationCriterion
from armscan_env.envs.observations import (
    ActionRewardObservation,
)
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.volumes.loading import RegisteredLabelmap
from armscan_env.wrapper import ArmscanEnvFactory

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import VectorEnvType
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    SACExperimentBuilder,
)
from tianshou.highlevel.params.alpha import AutoAlphaFactoryDefault
from tianshou.highlevel.params.policy_params import SACParams
from tianshou.utils.logging import datetime_tag

if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)

    volume_1 = RegisteredLabelmap.v1.load_labelmap()
    volume_2 = RegisteredLabelmap.v2.load_labelmap()

    log_name = os.path.join("sac", str(ExperimentConfig.seed), datetime_tag())
    experiment_config = ExperimentConfig()

    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=1000000,
        num_train_envs=40,
        num_test_envs=1,
        buffer_size=1000000,
        batch_size=256,
        step_per_collect=200,
        update_per_step=10,
        start_timesteps=5000,
        start_timesteps_random=True,
    )

    volume_size = volume_1.GetSize()
    env_factory = ArmscanEnvFactory(
        name2volume={"1": volume_1, "2": volume_2},
        observation=ActionRewardObservation(action_shape=(1,)).to_array_observation(),
        slice_shape=(volume_size[0], volume_size[2]),
        max_episode_len=10,
        rotation_bounds=(90.0, 45.0),
        translation_bounds=(0.0, None),
        seed=experiment_config.seed,
        venv_type=VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
        n_stack=3,
        termination_criterion=LabelmapEnvTerminationCriterion(min_reward_threshold=-0.1),
        reward_metric=LabelmapClusteringBasedReward(),
        project_actions_to="y",
    )

    experiment = (
        SACExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_sac_params(
            SACParams(
                tau=0.005,
                gamma=0.99,
                alpha=AutoAlphaFactoryDefault(lr=3e-4),
                estimation_step=1,
                actor_lr=1e-3,
                critic1_lr=1e-3,
                critic2_lr=1e-3,
            ),
        )
        .build()
    )

    experiment.run(run_name=log_name)
