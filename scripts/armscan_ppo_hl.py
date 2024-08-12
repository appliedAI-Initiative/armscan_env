import logging
import os

from armscan_env.config import get_config
from armscan_env.envs.labelmaps_navigation import LabelmapEnvTerminationCriterion
from armscan_env.envs.observations import (
    LabelmapSliceAsChannelsObservation,
)
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.network import ActorFactoryArmscanNet
from armscan_env.volumes.loading import RegisteredLabelmap
from armscan_env.wrapper import ArmscanEnvFactory

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import VectorEnvType
from tianshou.highlevel.experiment import ExperimentConfig, PPOExperimentBuilder
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactoryIndependentGaussians,
)
from tianshou.highlevel.params.policy_params import PPOParams
from tianshou.utils.logging import datetime_tag

if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)

    volume_1 = RegisteredLabelmap.v1.load_labelmap()
    volume_2 = RegisteredLabelmap.v2.load_labelmap()

    log_name = os.path.join(
        "ppo",
        str(ExperimentConfig.seed),
        "4_stack-lin_sweep_v1",
        datetime_tag(),
    )
    experiment_config = ExperimentConfig()

    sampling_config = SamplingConfig(
        num_epochs=500,
        step_per_epoch=10000,
        batch_size=80,
        num_train_envs=-1,
        num_test_envs=1,
        num_test_episodes=1,
        buffer_size=1600,
        step_per_collect=800,
    )

    volume_size = volume_1.GetSize()
    env_factory = ArmscanEnvFactory(
        name2volume={"1": volume_1},
        observation=LabelmapSliceAsChannelsObservation(
            slice_shape=(volume_size[0], volume_size[2]),
            action_shape=(4,),
        ),
        slice_shape=(volume_size[0], volume_size[2]),
        max_episode_len=20,
        rotation_bounds=(90.0, 45.0),
        translation_bounds=(0.0, None),
        render_mode="animation",
        seed=experiment_config.seed,
        venv_type=VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
        n_stack=4,
        termination_criterion=LabelmapEnvTerminationCriterion(min_reward_threshold=-0.1),
        reward_metric=LabelmapClusteringBasedReward(),
    )

    builder = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(
            PPOParams(
                discount_factor=0.99,
                gae_lambda=0.95,
                action_bound_method="clip",
                reward_normalization=False,
                ent_coef=0.01,
                vf_coef=0.25,
                max_grad_norm=0.5,
                value_clip=False,
                advantage_normalization=True,
                eps_clip=0.2,
                dual_clip=None,
                recompute_advantage=False,
                lr=2.5e-4,
                dist_fn=DistributionFunctionFactoryIndependentGaussians(),
            ),
        )
        .with_actor_factory(ActorFactoryArmscanNet())
        .with_critic_factory_use_actor()
    )
    experiment = builder.build()

    experiment.run(log_name)
