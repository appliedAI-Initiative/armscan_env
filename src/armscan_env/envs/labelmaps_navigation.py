import logging
from abc import ABC
from copy import copy, deepcopy
from typing import Any, ClassVar, Literal

import numpy as np
import SimpleITK as sitk
from armscan_env.clustering import TissueClusters
from armscan_env.constants import DEFAULT_SEED
from armscan_env.envs.base import (
    ModularEnv,
    Observation,
    RewardMetric,
    TerminationCriterion,
)
from armscan_env.envs.rewards import LabelmapClusteringBasedReward
from armscan_env.envs.state_action import LabelmapStateAction, ManipulatorAction
from armscan_env.util.visualizations import show_clusters
from armscan_env.volumes.slicing import (
    create_transformed_volume,
    get_volume_slice,
)
from celluloid import Camera
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gymnasium import logger as gym_logger  # type: ignore[attr-defined]
from gymnasium.core import ObsType as TObs
from gymnasium.spaces import Box, Space

log = logging.getLogger(__name__)

VOL_NAME_TO_OPTIMAL_ACTION = {
    "1": ManipulatorAction(rotation=(19.3, 0.0), translation=(0.0, 140.0)),
    "2": ManipulatorAction(rotation=(5, 0), translation=(0, 112)),
}


class LabelmapEnvTerminationCriterion(TerminationCriterion["LabelmapEnv"], ABC):
    def __init__(
        self,
        min_reward_threshold: float = -0.1,
    ):
        self.min_reward_threshold = min_reward_threshold

    def should_terminate(self, env: "LabelmapEnv") -> bool:
        return env.cur_reward > self.min_reward_threshold


class LabelmapEnv(ModularEnv[LabelmapStateAction, np.ndarray, np.ndarray]):
    """:param name2volume: mapping from labelmap names to volumes. One of these volumes will be selected at reset.
    :param observation: defines the observation space, e.g. `LabelmapSliceObservation`
    :param reward_metric: defines the reward metric that will be used, e.g. `LabelmapClusteringBasedReward`
    :param termination_criterion: if None, no termination criterion will be used
    :param slice_shape: determines the shape of the 2D slices that will be used as observations
    :param max_episode_len: maximum number of steps in an episode
    :param rotation_bounds: bounds for the rotation angles in degrees
    :param translation_bounds: bounds for the translation in mm. If None, the bound will be computed from the volume size.
    :param render_mode: determines how the environment will be rendered. Allowed values: "plt", "animation"
    :param seed: seed for the random number generator
    """

    _INITIAL_FULL_NORMALIZED_ACTION_ARR = np.zeros(4)

    metadata: ClassVar[dict] = {"render_modes": ["plt", "animation", None], "render_fps": 10}

    def __init__(
        self,
        name2volume: dict[str, sitk.Image],
        observation: Observation[LabelmapStateAction, Any],
        reward_metric: RewardMetric[LabelmapStateAction] = LabelmapClusteringBasedReward(),
        termination_criterion: TerminationCriterion | None = LabelmapEnvTerminationCriterion(),
        slice_shape: tuple[int, int] | None = None,
        max_episode_len: int | None = None,
        rotation_bounds: tuple[float, float] = (180, 180),
        translation_bounds: tuple[float | None, float | None] = (None, None),
        render_mode: Literal["plt", "animation"] | None = None,
        seed: int | None = DEFAULT_SEED,
        project_actions_to: Literal["x", "y", "xy"] | None = None,
        apply_volume_transformation: bool = False,
    ):
        if not name2volume:
            raise ValueError("name2volume must not be empty")
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unknown {render_mode=}. Allowed values: {self.metadata['render_modes']}",
            )

        super().__init__(reward_metric, observation, termination_criterion, max_episode_len)
        self.name2volume = name2volume
        self._slice_shape = slice_shape
        self._seed = seed
        self._project_actions_to = project_actions_to
        self._apply_volume_transformation = apply_volume_transformation

        # set at reset
        self._cur_labelmap_name: str | None = None
        self._cur_labelmap_volume: sitk.Image | None = None
        self._cur_optimal_action: ManipulatorAction | None = None

        self.user_defined_bounds = rotation_bounds, translation_bounds
        self.rotation_bounds = rotation_bounds
        self.translation_bounds = translation_bounds

        self.render_mode = render_mode
        self._fig: Figure | None = None
        self._axes: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes] | None = None
        self._camera: Camera | None = None

        # Safety measure for things like LabelmapClusterObservation
        # TODO: not nice, maybe make an interface
        if (
            hasattr(self.observation, "action_shape")
            and self.action_space.shape != self.observation.action_shape
        ):
            raise ValueError(
                f"The action space of the observation {self.observation} is {self.observation.action_shape} "
                f"and does not match the action space of the environment: {self.action_space}.",
            )

    @property
    def project_actions_to(self) -> Literal["x", "y", "xy"] | None:
        return self._project_actions_to

    def get_optimal_action(self) -> ManipulatorAction:
        if self.cur_optimal_action is None:
            raise RuntimeError("The optimal action must not be None, did you call reset?")
        return self.cur_optimal_action

    def get_full_optimal_action_array(self) -> np.ndarray:
        return self.get_optimal_action().to_normalized_array(
            self.rotation_bounds,
            self.translation_bounds,
        )

    def _get_projected_action_arr_idx(self) -> list[int]:
        match self._project_actions_to:
            case None:
                return list(range(4))
            case "x":
                return [2]
            case "y":
                return [3]
            case "xy":
                return [2, 3]
            case _:
                raise ValueError(f"Unknown {self._project_actions_to=}")

    def _get_full_action_arr_len(self) -> int:
        return len(self._INITIAL_FULL_NORMALIZED_ACTION_ARR)

    def _get_full_action_leading_to_initial_state_normalized_arr(self) -> np.ndarray:
        initial_action_arr = self.get_full_optimal_action_array()
        project_idx = self._get_projected_action_arr_idx()
        initial_action_arr[project_idx] = copy(
            self._INITIAL_FULL_NORMALIZED_ACTION_ARR[project_idx],
        )
        return initial_action_arr

    def _get_action_leading_to_initial_state(self) -> ManipulatorAction:
        return ManipulatorAction.from_normalized_array(
            self._get_full_action_leading_to_initial_state_normalized_arr(),
            self.rotation_bounds,
            self.translation_bounds,
        )

    def get_optimal_action_array(self) -> np.ndarray:
        full_action_arr = self.get_full_optimal_action_array()
        return full_action_arr[self._get_projected_action_arr_idx()]

    def step_to_optimal_state(
        self,
    ) -> tuple[Observation[LabelmapStateAction, Any], float, bool, bool, dict[str, Any]]:
        return self.step(self.get_optimal_action())

    @property
    def cur_labelmap_name(self) -> str | None:
        return self._cur_labelmap_name

    @property
    def cur_labelmap_volume(self) -> sitk.Image | None:
        return self._cur_labelmap_volume

    @property
    def cur_optimal_action(self) -> ManipulatorAction | None:
        return self._cur_optimal_action

    @property
    def action_space(self) -> Space[np.ndarray]:
        action_dim = len(self._get_projected_action_arr_idx())
        return Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    def close(self) -> None:
        super().close()
        self.name2volume = {}
        self._cur_labelmap_name = None
        self._cur_labelmap_volume = None
        self._cur_optimal_action = None

    def get_full_action_array_from_projected_action(
        self,
        normalized_action_arr: np.ndarray,
    ) -> np.ndarray:
        """Converts a (potentially projected and) normalized action array to a full action array.

        If `project_actions_to` is not None, the `normalized_action_arr` is assumed to be a projection
        of the correct dimension.
        """
        full_action_arr = self.get_full_optimal_action_array()
        project_idx = self._get_projected_action_arr_idx()
        if len(normalized_action_arr) != len(project_idx):
            raise ValueError(
                f"Expected {len(project_idx)} elements in normalized_action_arr, "
                f"but got {len(normalized_action_arr)}",
            )
        full_action_arr[project_idx] = normalized_action_arr
        return full_action_arr

    def get_manipulator_action_from_normalized_action(
        self,
        normalized_action_arr: np.ndarray,
    ) -> ManipulatorAction:
        """Converts a (potentially projected and) normalized action array to a ManipulatorAction.

        Passing a full action array is also supported, even if `project_actions_to` is not None.
        If `normalized_action_arr` is of a lower len and `project_actions_to` is not None, the
        `normalized_action_arr` is assumed to be a projection
        of the correct dimension.
        """
        if len(normalized_action_arr) != self._get_full_action_arr_len():
            normalized_action_arr = self.get_full_action_array_from_projected_action(
                normalized_action_arr,
            )
        return ManipulatorAction.from_normalized_array(
            normalized_action_arr,
            self.rotation_bounds,
            self.translation_bounds,
        )

    def _get_slice_from_action(self, action: np.ndarray | ManipulatorAction) -> np.ndarray:
        if isinstance(action, np.ndarray):
            manipulator_action = self.get_manipulator_action_from_normalized_action(action)
        else:
            manipulator_action = action
        sliced_volume = get_volume_slice(
            volume=self.cur_labelmap_volume,
            slice_shape=self._slice_shape,
            action=manipulator_action,
        )
        return sitk.GetArrayFromImage(sliced_volume).T

    def _get_initial_slice(self) -> np.ndarray:
        action_to_initial_slice = self._get_action_leading_to_initial_state()
        return self._get_slice_from_action(action_to_initial_slice)

    def compute_next_state(
        self,
        normalized_action_arr: np.ndarray | ManipulatorAction,
    ) -> LabelmapStateAction:
        new_slice = self._get_slice_from_action(normalized_action_arr)
        return LabelmapStateAction(
            normalized_action_arr=normalized_action_arr,
            labels_2d_slice=new_slice,
            # cur_state_action is the previous state, so this reward is computed for the previous state
            optimal_position=self.cur_state_action.optimal_position,
            optimal_labelmap=self.cur_state_action.optimal_labelmap,
        )

    def apply_volume_transformation(
        self,
        volume: sitk.Image,
        volume_transformation_action: ManipulatorAction,
        optimal_action: ManipulatorAction,
    ) -> (sitk.Image, ManipulatorAction):  # type: ignore
        """Apply a random transformation to the volume and to the optimal action. The transformation is a random rotation
        and translation. The bounds of the rotation are updated if they have already been set. The translation bounds are
        computed from the volume size in the 'sample_initial_state' method.

        :param volume: the volume to transform
        :param volume_transformation_action: the transformation action to apply to the volume
        :param optimal_action: the optimal action for the volume to transform accordingly
        :return: the transformed volume and the transformed optimal action
        """
        transformed_volume = create_transformed_volume(
            volume=volume,
            transformation_action=volume_transformation_action,
        )
        transformed_optimal_action = transformed_volume.transform_action(
            optimal_action,
        )
        if self.rotation_bounds:
            bounds = list(self.rotation_bounds)
            bounds[0] += abs(volume_transformation_action.rotation[0])
            bounds[1] += abs(volume_transformation_action.rotation[1])
            self.rotation_bounds = tuple(bounds)  # type: ignore
        return (
            transformed_volume,
            transformed_optimal_action,
        )

    def sample_initial_state(self) -> LabelmapStateAction:
        if not self.name2volume:
            raise RuntimeError(
                "The name2volume attribute must not be empty. Did you close the environment?",
            )
        sampled_image_name = np.random.choice(list(self.name2volume.keys()))
        self._cur_labelmap_name = sampled_image_name
        volume_optimal_action = deepcopy(VOL_NAME_TO_OPTIMAL_ACTION[sampled_image_name])

        if self._apply_volume_transformation:
            volume_transformation_action = ManipulatorAction.sample()
            self._cur_labelmap_volume, self._cur_optimal_action = self.apply_volume_transformation(
                volume=self.name2volume[sampled_image_name],
                volume_transformation_action=volume_transformation_action,
                optimal_action=volume_optimal_action,
            )
        else:
            self._cur_labelmap_volume = self.name2volume[sampled_image_name]
            self._cur_optimal_action = volume_optimal_action
        if None in self.translation_bounds:
            self._compute_translation_bounds()
        if self._slice_shape is None:
            self.compute_slice_shape(volume=self.cur_labelmap_volume)
        initial_slice = self._get_initial_slice()
        return LabelmapStateAction(
            normalized_action_arr=copy(
                self._INITIAL_FULL_NORMALIZED_ACTION_ARR[self._get_projected_action_arr_idx()],
            ),
            labels_2d_slice=initial_slice,
            # TODO: pass the env's optimal position and labelmap or remove them from the StateAction?
            optimal_position=self._cur_optimal_action,
            optimal_labelmap=None,
        )

    def compute_slice_shape(self, volume: sitk.Image | None) -> None:
        """Compute the shape of the 2D slices that will be used as observations."""
        if volume is None:
            raise RuntimeError("The labelmap volume must not be None, did you call reset?")
        size = volume.GetSize()
        self._slice_shape = (size[0], size[2])

    def _compute_translation_bounds(self) -> None:
        """Compute the translation bounds from the volume size."""
        if self.cur_labelmap_volume is None:
            raise RuntimeError("The labelmap volume must not be None, did you call reset?")
        volume = self.cur_labelmap_volume
        size = volume.GetSize()
        spacing = volume.GetSpacing()
        bounds = list(self.translation_bounds)
        for i in range(2):
            if bounds[i] is None:
                bounds[i] = size[i] * spacing[i]
        self.translation_bounds = tuple(bounds)  # type: ignore

    def get_cur_full_normalized_action_arr(self) -> np.ndarray:
        return self.get_full_action_array_from_projected_action(
            self.cur_state_action.normalized_action_arr,
        )

    def get_cur_manipulator_action(self) -> ManipulatorAction:
        cur_full_action_arr = self.get_cur_full_normalized_action_arr()
        return ManipulatorAction.from_normalized_array(
            cur_full_action_arr,
            self.rotation_bounds,
            self.translation_bounds,
        )

    def step(
        self,
        action: np.ndarray | ManipulatorAction,
    ) -> tuple[Observation[LabelmapStateAction, Any], float, bool, bool, dict[str, Any]]:
        if isinstance(action, ManipulatorAction):
            action = action.to_normalized_array(self.rotation_bounds, self.translation_bounds)
        return super().step(action)

    def reset(
        self,
        reset_render: bool = True,
        reset_translation_bounds: bool = True,
        reset_slice_shape: bool = False,
        **kwargs: Any,
    ) -> tuple[TObs, dict[str, Any]]:
        if reset_render:
            self.reset_render()
        if reset_translation_bounds:
            self.translation_bounds = self.user_defined_bounds[1]
        if reset_slice_shape:
            self._slice_shape = None
        # Remove the seed from kwargs if it exists
        kwargs.pop("seed", None)
        obs, info = super().reset(seed=self._seed, **kwargs)
        return obs, info

    def render(self, title: str = "Labelmap slice") -> plt.Figure | Camera | None:
        match self.render_mode:
            case "plt":
                return self.get_cur_state_plot(create_new_figure=False, title=title)
            case "animation":
                camera = self.get_camera()
                self.get_cur_state_plot(create_new_figure=False, title=title)
                camera.snap()
                return camera
            case None:
                gym_logger.warn(
                    "You are calling render method without having specified any render mode. "
                    "You can specify the render_mode at initialization, "
                    f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")',
                )
                return None
            case _:
                raise RuntimeError(
                    f"Unknown render mode: {self.render_mode}, this should not happen.",
                )

    def get_cur_state_plot(
        self,
        create_new_figure: bool = True,
        title: str = "Labelmap slice",
    ) -> plt.Figure | None:
        """Retrieve a figure visualizing the current state of the environment.

        :param create_new_figure: if True, a new figure will be created. Otherwise, a single figure will be used
            in subsequent calls to this method. False may be useful for animations.
        """
        fig, (ax1, ax2, ax3, ax4, ax5) = self.get_figure_axes()  # type: ignore
        if create_new_figure:
            fig, ax1, ax2, ax3, ax4, ax5 = copy((fig, ax1, ax2, ax3, ax4, ax5))

        if self.cur_labelmap_volume is None:
            raise RuntimeError("The labelmap volume must not be None, did you call reset?")

        volume = self.cur_labelmap_volume
        o = volume.GetOrigin()
        img_array = sitk.GetArrayFromImage(volume)
        action = self.get_manipulator_action_from_normalized_action(
            self.cur_state_action.normalized_action_arr,
        )
        translation = action.translation
        rotation = action.rotation

        # Subplot 1: from the top
        iz = volume.GetSize()[2] // 2
        ax1.imshow(img_array[iz, :, :])
        x_dash = np.arange(img_array.shape[2])
        b = volume.TransformPhysicalPointToIndex(
            [o[0] + translation[0], o[1] + translation[1], o[2]],
        )[1]
        b_x = b + np.tan(np.deg2rad(rotation[1])) * iz
        y_dash = np.tan(np.deg2rad(rotation[0])) * x_dash + b_x
        y_dash = np.clip(y_dash, 0, img_array.shape[1] - 1)
        ax1.plot(x_dash, y_dash, linestyle="--", color="red")
        ax1.set_title(f"Slice cut (labelmap name: {self.cur_labelmap_name})")

        # Subplot 2: from the side
        ix = volume.GetSize()[0] // 2
        ax2.imshow(img_array[:, :, ix].T)
        z_dash = np.arange(img_array.shape[0])
        b_z = b + np.tan(np.deg2rad(rotation[0])) * ix
        y_dash_2 = np.tan(np.deg2rad(rotation[1])) * z_dash + b_z
        y_dash_2 = np.clip(y_dash_2, 0, img_array.shape[1] - 1)
        ax2.plot(z_dash, y_dash_2, linestyle="--", color="red")

        # ACTION
        sliced_img = self.cur_state_action.labels_2d_slice
        ax3.imshow(sliced_img.T, origin="lower", aspect=6)

        txt = (
            "Slice taken at position:\n"
            f"y: {translation[1]:.2f} mm,\n"
            f"x: {translation[0]:.2f} mm,\n"
            f"rot_z: {rotation[0]:.2f} deg,\n"
            f"rot_x: {rotation[1]:.2f} deg\n"
            f"terminated: {self.is_terminated}, truncated: {self.is_truncated}\n"
        )
        ax4.text(0.5, 0.5, txt, ha="center", va="center")
        ax4.axis("off")
        ax4.text(0.5, 0.8, f"Step: {self.cur_episode_len}", ha="center", va="center")

        # OBSERVATION
        clusters = TissueClusters.from_labelmap_slice(self.cur_state_action.labels_2d_slice)
        show_clusters(clusters, sliced_img, ax5)

        # REWARD
        ax5.text(0, 0, f"Reward: {self.cur_reward:.2f}", fontsize=12, color="red")

        if fig is not None:
            fig.suptitle(title, x=0.2, y=0.95)

        plt.close()
        return fig

    def _create_figure_axis(self) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(12, 9))
        gs = fig.add_gridspec(nrows=3, ncols=3)

        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[2, 2])

        self._fig = fig
        self._axes = (ax1, ax2, ax3, ax4, ax5)

    def get_figure_axes(self) -> tuple[Figure | None, tuple[Axes, Axes, Axes, Axes, Axes] | None]:
        if self._fig is None or self._axes is None:
            self._create_figure_axis()
        if self._fig is None or self._axes is None:
            raise RuntimeError("Could not create figure and axes.")
        return self._fig, self._axes

    def get_camera(self) -> Camera:
        if self._camera is None:
            fig, ax = self.get_figure_axes()
            self._camera = Camera(fig)
        return self._camera

    def get_cur_animation(self) -> ArtistAnimation:
        return self.get_camera().animate()

    def get_cur_animation_as_html(self) -> HTML:
        anim = self.get_cur_animation()
        return HTML(anim.to_jshtml())

    def reset_render(self) -> None:
        self._camera = None
        self._fig = None
        self._axes = None
