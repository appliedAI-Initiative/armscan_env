# %%

import os

import SimpleITK as sitk
from armscan_env.envs.labelmaps_navigation import (
    LabelmapSliceAsChannelsObservation,
)
from armscan_env.wrapper import ActorFactoryArmscanDQN, ArmscanEnvFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import ExperimentConfig, PPOExperimentBuilder
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactoryIndependentGaussians,
)
from tianshou.highlevel.params.policy_params import PPOParams
from tianshou.utils.logging import datetime_tag

# %%
path_to_labels_1 = os.path.join("..", "data", "labels", "00001_labels.nii")
volume_1 = sitk.ReadImage(path_to_labels_1)
path_to_labels_2 = os.path.join("..", "data", "labels", "00002_labels.nii")
volume_2 = sitk.ReadImage(path_to_labels_2)

log_name = os.path.join("ppo", str(ExperimentConfig.seed), datetime_tag())
experiment_config = ExperimentConfig()

# %%
sampling_config = SamplingConfig(
    num_epochs=100,
    step_per_epoch=1000,
    batch_size=25,
    num_train_envs=-1,
    num_test_envs=10,
    buffer_size=100000,
    step_per_collect=1000,
    replay_buffer_stack_num=4,
)

volume_size = volume_1.GetSize()
env_factory = ArmscanEnvFactory(
    name2volume={"1": volume_1, "2": volume_2},
    observation=LabelmapSliceAsChannelsObservation(slice_shape=(volume_size[2], volume_size[0])),
    slice_shape=(volume_size[0], volume_size[2]),
    max_episode_len=5000,
    angle_bounds=(90.0, 45.0),
    translation_bounds=(0.0, None),
    render_mode="animation",
    seed=experiment_config.seed,
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
    .with_actor_factory(ActorFactoryArmscanDQN(features_only=True, output_dim_added_layer=(512)))
    .with_critic_factory_use_actor()
)
experiment = builder.build()

# %%
experiment.run(log_name)
