from pathlib import Path
from typing import cast

from armscan_env.wrapper import ArmscanEnvFactory

from tianshou.highlevel.env import EnvMode
from tianshou.highlevel.experiment import Experiment

# Place your path here
saved_experiment_dir = Path("log/random-actions/42/20240814-150513")

if __name__ == "__main__":
    restored_experiment = Experiment.from_directory(str(saved_experiment_dir), restore_policy=True)

    restored_policy = restored_experiment.create_experiment_world().policy
    # this can be now used to perform actions, you can also use a notebook

    env_factory: ArmscanEnvFactory = cast(ArmscanEnvFactory, restored_experiment.env_factory)
    # TODO @Carlo: modify here to set different volumes
    env_factory.name2volume = env_factory.name2volume

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
