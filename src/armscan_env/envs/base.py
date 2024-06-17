from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

import gymnasium as gym

TObs = TypeVar("TObs")
TAction = TypeVar("TAction")


class EnvPreconditionError(RuntimeError):
    pass


@dataclass(kw_only=True)
class StateAction:
    normalized_action_arr: Any
    # state of the env will be reflected by fields added to subclasses
    # but action is a reserved field name. Subclasses should override the
    # type of action to be more specific


TStateAction = TypeVar("TStateAction", bound=StateAction)
TEnv = TypeVar("TEnv", bound="ModularEnv")


class RewardMetric(Generic[TStateAction], ABC):
    @abstractmethod
    def compute_reward(self, state: TStateAction) -> float:
        pass

    @property
    @abstractmethod
    def range(self) -> tuple[float, float]:
        pass


class TerminationCriterion(Generic[TEnv], ABC):
    @abstractmethod
    def should_terminate(self, env: TEnv) -> bool:
        pass


class NeverTerminate(TerminationCriterion[Any]):
    def should_terminate(self, env: Any) -> bool:
        return False


class Observation(Generic[TStateAction, TObs], ABC):
    @abstractmethod
    def compute_observation(self, state: TStateAction) -> TObs:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space[TObs]:
        pass


class DictObservation(Observation[TStateAction, dict[str, np.ndarray]], Generic[TStateAction], ABC):
    pass


class ArrayObservation(Observation[TStateAction, np.ndarray], Generic[TStateAction], ABC):
    pass


@dataclass(kw_only=True)
class EnvStatus(Generic[TStateAction, TObs]):
    episode_len: int
    state_action: TStateAction | None
    observation: TObs | None
    reward: float | None
    is_terminated: bool
    is_truncated: bool
    is_closed: bool
    info: dict[str, Any]


class ModularEnv(gym.core.Env[TObs, TAction], Generic[TStateAction, TAction, TObs], ABC):
    def __init__(
        self,
        reward_metric: RewardMetric[TStateAction],
        observation: Observation[TStateAction, TObs],
        termination_criterion: TerminationCriterion | None = None,
        max_episode_len: int | None = None,
    ):
        self.reward_metric = reward_metric
        self.observation = observation
        self.termination_criterion = termination_criterion or NeverTerminate()
        self.max_episode_len = max_episode_len

        self._is_closed = True
        self._is_terminated = False
        self._is_truncated = False
        self._cur_episode_len = 0
        self._cur_observation: TObs | None = None
        self._cur_reward: float | None = None
        self._cur_state_action: TStateAction | None = None

    def get_cur_env_status(self) -> EnvStatus[TStateAction, TObs]:
        return EnvStatus(
            episode_len=self.cur_episode_len,
            state_action=self.cur_state_action,
            observation=self.cur_observation,
            reward=self.cur_reward,
            is_terminated=self.is_terminated,
            is_truncated=self._is_truncated,
            is_closed=self.is_closed,
            info=self.get_info_dict(),
        )

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def is_terminated(self) -> bool:
        return self._is_terminated

    @property
    def is_truncated(self) -> bool:
        return self._is_truncated

    @property
    def cur_state_action(self) -> TStateAction | None:
        return self._cur_state_action

    @property
    def cur_observation(self) -> TObs | None:
        return self._cur_observation

    @property
    def cur_reward(self) -> float | None:
        return self._cur_reward

    @property
    def cur_episode_len(self) -> int:
        return self._cur_episode_len

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space[TAction]:
        pass

    @property
    def observation_space(self) -> gym.spaces.Space[TObs]:
        return self.observation.observation_space

    def close(self) -> None:
        self._cur_state_action = None
        self._is_closed = True
        self._cur_episode_len = 0

    def _assert_cur_state(self) -> None:
        if self.cur_state_action is None:
            raise EnvPreconditionError(
                "This operation requires a current state, but none is set. Did you call reset()?",
            )

    @abstractmethod
    def compute_next_state(self, action: TAction) -> TStateAction:
        pass

    @abstractmethod
    def sample_initial_state(self) -> TStateAction:
        pass

    def get_info_dict(self) -> dict[str, Any]:
        # override this if you want to return additional info
        return {}

    def should_terminate(self) -> bool:
        return self.termination_criterion.should_terminate(self)

    def should_truncate(self) -> bool:
        if self.max_episode_len is not None:
            return self.cur_episode_len >= self.max_episode_len
        return False

    def compute_cur_observation(self) -> TObs:
        self._assert_cur_state()
        assert self.cur_state_action is not None
        return self.observation.compute_observation(self.cur_state_action)

    def compute_cur_reward(self) -> float:
        self._assert_cur_state()
        assert self.cur_state_action is not None
        return self.reward_metric.compute_reward(self.cur_state_action)

    def _update_cur_reward(self) -> None:
        self._cur_reward = self.compute_cur_reward()

    def _update_cur_observation(self) -> None:
        self._cur_observation = self.compute_cur_observation()

    def _update_is_terminated(self) -> None:
        self._is_terminated = self.should_terminate()

    def _update_is_truncated(self) -> None:
        self._is_truncated = self.should_truncate()

    def _update_observation_reward_termination(self) -> None:
        # NOTE: the order of these calls is important!
        self._update_cur_observation()
        self._update_cur_reward()
        self._update_is_terminated()
        self._update_is_truncated()

    def _go_to_next_state(self, action: TAction) -> None:
        self._cur_state_action = self.compute_next_state(action)
        self._update_observation_reward_termination()

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[TObs, dict[str, Any]]:
        super().reset(seed=seed, **kwargs)
        self._cur_state_action = self.sample_initial_state()
        self._is_closed = False
        self._cur_episode_len = 0
        self._update_observation_reward_termination()
        assert self.cur_observation is not None
        return self.cur_observation, self.get_info_dict()

    def step(self, action: TAction) -> tuple[TObs, float, bool, bool, dict[str, Any]]:
        """Step through the environment to navigate to the next state."""
        self._go_to_next_state(action)
        self._cur_episode_len += 1
        assert self.cur_observation is not None
        assert self.cur_reward is not None
        return (
            self.cur_observation,
            self.cur_reward,
            self.is_terminated,
            self.is_truncated,
            self.get_info_dict(),
        )


@dataclass(kw_only=True)
class EnvRollout(Generic[TObs, TAction]):
    observations: list[TObs] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    actions: list[TAction | None] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)

    def append_step(
        self,
        action: TAction,
        observation: TObs,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(action)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.infos.append(info)

    def append_reset(
        self,
        observation: TObs,
        info: dict[str, Any],
        reward: float = 0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(None)
        self.infos.append(info)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
