import logging
import os

import SimpleITK as sitk
from armscan_env.config import get_config
from armscan_env.envs.labelmaps_navigation import LabelmapEnvTerminationCriterion
from armscan_env.envs.observations import (
    ActionRewardObservation,
    LabelmapClusterObservation,
)
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
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

    volume_1 = sitk.ReadImage(config.get_labels_path(1))
    volume_2 = sitk.ReadImage(config.get_labels_path(2))

    log_name = os.path.join("sac-characteristic-array", str(ExperimentConfig.seed), datetime_tag())
    experiment_config = ExperimentConfig()

    sampling_config = SamplingConfig(
        num_epochs=10,
        step_per_epoch=100,
        num_train_envs=1,
        num_test_envs=1,
        buffer_size=100,
        batch_size=256,
        step_per_collect=200,
        update_per_step=2,
        start_timesteps=50,
        start_timesteps_random=True,
    )

    volume_size = volume_1.GetSize()
    env_factory = ArmscanEnvFactory(
        name2volume={"1": volume_1},
        observation=LabelmapClusterObservation()
        .merged_with(other=ActionRewardObservation(action_shape=(1,)))  # type: ignore
        .to_array_observation(),
        slice_shape=(volume_size[0], volume_size[2]),
        max_episode_len=20,
        rotation_bounds=(90.0, 45.0),
        translation_bounds=(0.0, None),
        render_mode="animation",
        seed=experiment_config.seed,
        venv_type=VectorEnvType.DUMMY,
        n_stack=2,
        termination_criterion=LabelmapEnvTerminationCriterion(min_reward_threshold=-0.1),
        reward_metric=LabelmapClusteringBasedReward(),
        project_actions_to="y",
        apply_volume_transformation=True,
        add_reward_details=True,
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
        .with_actor_factory_default(
            (256, 256),
            continuous_unbounded=True,
            continuous_conditioned_sigma=True,
        )
        .with_common_critic_factory_default((256, 256))
        .build()
    )

    experiment.run(run_name=log_name)
