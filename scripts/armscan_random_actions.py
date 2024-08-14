import logging
import os

from armscan_env.envs.base import DummyArrayObservation
from armscan_env.envs.labelmaps_navigation import LabelmapEnvTerminationCriterion
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.volumes.loading import load_sitk_volumes
from armscan_env.wrapper import ArmscanEnvFactory
from sensai.util.logging import datetime_tag

from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import VectorEnvType
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    RandomActionExperimentBuilder,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    volumes = load_sitk_volumes()
    n_random_steps = 2000
    seed = 42

    log_name = os.path.join(
        "random-actions",
        str(ExperimentConfig.seed),
        datetime_tag(),
    )
    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=n_random_steps,
        step_per_collect=500,
    )

    volume_size = volumes[0].GetSize()
    env_factory = ArmscanEnvFactory(
        name2volume={
            "1": volumes[0],
            "2": volumes[1],
            "3": volumes[2],
            "4": volumes[3],
            "5": volumes[4],
            "6": volumes[5],
            "7": volumes[6],
            "8": volumes[7],
        },
        observation=DummyArrayObservation(),
        slice_shape=(volume_size[0], volume_size[2]),
        max_episode_len=50,
        rotation_bounds=(90.0, 45.0),
        translation_bounds=(None, None),
        seed=seed,
        venv_type=VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
        termination_criterion=LabelmapEnvTerminationCriterion(min_reward_threshold=-0.05),
        reward_metric=LabelmapClusteringBasedReward(),
        project_actions_to="zy",
        apply_volume_transformation=True,
        best_reward_memory=0,
    )
    experiment = RandomActionExperimentBuilder(env_factory, sampling_config=sampling_config).build()

    experiment.run(run_name=log_name)
