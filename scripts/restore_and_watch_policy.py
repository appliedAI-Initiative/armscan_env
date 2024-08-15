from pathlib import Path
from typing import cast, Sequence

from armscan_env.volumes.loading import load_sitk_volumes
from armscan_env.wrapper import ArmscanEnvFactory

from tianshou.highlevel.env import EnvMode
from tianshou.highlevel.experiment import Experiment

# Place your path here
saved_experiment_dir = Path("log/sac-characteristic-array-rew-details-y/42/20240630-191219")

if __name__ == "__main__":
    volumes = load_sitk_volumes()
    restored_experiment = Experiment.from_directory(str(saved_experiment_dir), restore_policy=True)

    restored_policy = restored_experiment.create_experiment_world().policy
    # this can be now used to perform actions, you can also use a notebook

    old_env_factory: ArmscanEnvFactory = cast(ArmscanEnvFactory, restored_experiment.env_factory)
    # TODO @Carlo: modify here to set different volumes
    env_factory = ArmscanEnvFactory(
        name2volume={"3": volumes[2]},
        observation=old_env_factory.observation,
        reward_metric=old_env_factory.reward_metric,
        termination_criterion=old_env_factory.termination_criterion,
        slice_shape=old_env_factory.slice_shape,
        max_episode_len=old_env_factory.max_episode_len,
        rotation_bounds=old_env_factory.rotation_bounds,
        translation_bounds=old_env_factory.translation_bounds,
        render_mode_train=old_env_factory.render_modes[EnvMode.TRAIN],
        render_mode_test=old_env_factory.render_modes[EnvMode.TEST],
        render_mode_watch=old_env_factory.render_modes[EnvMode.WATCH],
        venv_type=old_env_factory.venv_type,
        seed=old_env_factory.seed,
        n_stack=old_env_factory.n_stack,
        project_actions_to=old_env_factory.project_actions_to,
        apply_volume_transformation=old_env_factory.apply_volume_transformation,
        best_reward_memory=0,
        exclude_keys_from_framestack=(),
        **old_env_factory.make_kwargs
    )

    # Create env manually and run policy on it
    restored_env = env_factory.create_env(mode=EnvMode.WATCH)
    obs, info = restored_env.reset()
    for _ in range(5):
        obs, *_ = restored_env.step(restored_policy.compute_action(obs))

    # Or use the restored experiment to run the policy
    restored_experiment.config.train = False
    restored_experiment.config.watch = True
    restored_experiment.config.persistence_enabled = False
    restored_experiment.run()
