import logging
from abc import ABC
from copy import copy
from typing import Any, ClassVar, Literal

import gymnasium as gym
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
from armscan_env.slicing import slice_volume
from armscan_env.util.visualizations import show_clusters
from celluloid import Camera
from gymnasium.core import ObsType as TObs
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


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
    :param observation: defines the observation space, e.g. LabelmapSliceObservation
    :param reward_metric: defines the reward metric that will be used, e.g. LabelmapClusteringBasedReward
    :param termination_criterion: if None, no termination criterion will be used
    :param slice_shape: determines the shape of the 2D slices that will be used as observations
    :param max_episode_len: maximum number of steps in an episode
    :param angle_bounds: bounds for the rotation angles in degrees
    :param translation_bounds: bounds for the translation in mm. If None, the bound will be computed from the volume size.
    :param render_mode: determines how the environment will be rendered. Allowed values: "plt", "animation"
    :param seed: seed for the random number generator
    """

    _INITIAL_POS_ROTATION = np.zeros(4)

    metadata: ClassVar[dict] = {"render_modes": ["plt", "animation", None], "render_fps": 10}

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
        render_mode: Literal["plt", "animation"] | None = None,
        seed: int | None = DEFAULT_SEED,
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

        # set at reset
        self._cur_labelmap_name: str | None = None
        self._cur_labelmap_volume: sitk.Image | None = None

        self.user_defined_bounds = angle_bounds, translation_bounds
        self.angle_bounds = angle_bounds
        self.translation_bounds = translation_bounds

        self.render_mode = render_mode
        self._fig: Figure | None = None
        self._axes: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes] | None = None
        self._camera: Camera | None = None

    def unnormalize_rotation_translation(self, action: np.ndarray) -> ManipulatorAction:
        """Unnormalizes an array with values in the range [-1, 1] to the original range that is
        consumed by :func:`slice_volume`.

        :param action: normalized action with values in the range [-1, 1]
        :return: unnormalized action that can be used with :func:`slice_volume`
        """
        return ManipulatorAction.from_normalized_array(
            action,
            self.angle_bounds,
            self.translation_bounds,
        )

    @property
    def cur_labelmap_name(self) -> str | None:
        return self._cur_labelmap_name

    @property
    def cur_labelmap_volume(self) -> sitk.Image | None:
        return self._cur_labelmap_volume

    @property
    def action_space(self) -> gym.spaces.Space[np.ndarray]:
        return gym.spaces.Box(
            low=-1,
            high=1.0,
            shape=(4,),
        )  # 2 rotations, 2 translations.

    def close(self) -> None:
        super().close()
        self.name2volume = {}
        self._cur_labelmap_name = None
        self._cur_labelmap_volume = None

    def _get_slice_from_action(self, action: np.ndarray) -> np.ndarray:
        manipulator_action = self.unnormalize_rotation_translation(action)
        sliced_volume = slice_volume(
            volume=self.cur_labelmap_volume,
            slice_shape=self._slice_shape,
            z_rotation=manipulator_action.rotation[0],
            x_rotation=manipulator_action.rotation[1],
            x_trans=manipulator_action.translation[0],
            y_trans=manipulator_action.translation[1],
        )
        return sitk.GetArrayFromImage(sliced_volume)[:, 0, :]

    def _get_initial_slice(self) -> np.ndarray:
        return self._get_slice_from_action(self._INITIAL_POS_ROTATION)

    def compute_next_state(self, action: np.ndarray) -> LabelmapStateAction:
        new_slice = self._get_slice_from_action(action)
        return LabelmapStateAction(
            action=action,
            labels_2d_slice=new_slice,
            last_reward=self.reward_metric.compute_reward(self.cur_state_action),
            # cur_state_action is the previous state, so this reward is computed for the previous state
            optimal_position=self.cur_state_action.optimal_position,
            optimal_labelmap=self.cur_state_action.optimal_labelmap,
        )

    def sample_initial_state(self) -> LabelmapStateAction:
        if not self.name2volume:
            raise RuntimeError(
                "The name2volume attribute must not be empty. Did you close the environment?",
            )
        sampled_image_name = np.random.choice(list(self.name2volume.keys()))
        self._cur_labelmap_name = sampled_image_name
        self._cur_labelmap_volume = self.name2volume[sampled_image_name]
        if None in self.translation_bounds:
            self.compute_translation_bounds()
        if self._slice_shape is None:
            self.compute_slice_shape(volume=self.cur_labelmap_volume)
        # Alternatively, select a random slice
        initial_slice = self._get_initial_slice()
        return LabelmapStateAction(
            action=self._INITIAL_POS_ROTATION,
            labels_2d_slice=initial_slice,
            last_reward=-1,
            optimal_position=None,
            optimal_labelmap=None,
        )

    def compute_slice_shape(self, volume: sitk.Image | None) -> None:
        """Compute the shape of the 2D slices that will be used as observations."""
        if volume is None:
            raise RuntimeError("The labelmap volume must not be None, did you call reset?")
        size = volume.GetSize()
        self._slice_shape = (size[0], size[2])

    def compute_translation_bounds(self) -> None:
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

    def get_translation_bounds(self) -> tuple[float | None, float | None]:
        return self.translation_bounds

    def step(
        self,
        action: np.ndarray | ManipulatorAction,
    ) -> tuple[TObs, float, bool, bool, dict[str, Any]]:
        if isinstance(action, ManipulatorAction):
            action = action.to_normalized_array(self.angle_bounds, self.translation_bounds)
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
        obs, info = super().reset(seed=self._seed, **kwargs)
        return obs, info

    def render(self) -> plt.Figure | Camera | None:
        match self.render_mode:
            case "plt":
                return self.get_cur_state_plot(create_new_figure=False)
            case "animation":
                camera = self.get_camera()
                self.get_cur_state_plot(create_new_figure=False)
                camera.snap()
                return camera
            case None:
                gym.logger.warn(
                    "You are calling render method without having specified any render mode. "
                    "You can specify the render_mode at initialization, "
                    f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")',
                )
                return None
            case _:
                raise RuntimeError(
                    f"Unknown render mode: {self.render_mode}, this should not happen.",
                )

    def get_cur_state_plot(self, create_new_figure: bool = True) -> plt.Figure | None:
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
        action = self.unnormalize_rotation_translation(self.cur_state_action.action)
        translation = action.translation
        rotation = action.rotation

        # Subplot 1: from the top
        iz = volume.GetSize()[2] // 2
        ax1.imshow(img_array[iz, :, :])
        x_dash = np.arange(img_array.shape[2])
        b = volume.TransformPhysicalPointToIndex([o[0], o[1] + translation[1], o[2]])[1]
        b_x = b + np.tan(np.deg2rad(rotation[1])) * iz
        y_dash = np.tan(np.deg2rad(rotation[0])) * x_dash + b_x
        y_dash = np.clip(y_dash, 0, img_array.shape[1] - 1)
        ax1.plot(x_dash, y_dash, linestyle="--", color="red")
        ax1.set_title("Slice cut")

        # Subplot 2: from the side
        ix = volume.GetSize()[0] // 2
        ax2.imshow(img_array[:, :, ix].T, aspect=0.24)
        z_dash = np.arange(img_array.shape[0])
        b_z = b + np.tan(np.deg2rad(rotation[0])) * ix
        y_dash_2 = np.tan(np.deg2rad(rotation[1])) * z_dash + b_z
        y_dash_2 = np.clip(y_dash_2, 0, img_array.shape[1] - 1)
        ax2.plot(z_dash, y_dash_2, linestyle="--", color="red")

        # ACTION
        sliced_img = self.cur_state_action.labels_2d_slice
        ax3.imshow(sliced_img, origin="lower", aspect=6)

        txt = (
            "Slice taken at position:\n"
            f"y: {translation[1]:.2f} mm,\n"
            f"x: {translation[0]:.2f} mm,\n"
            f"rot_z: {rotation[0]:.2f} deg,\n"
            f"rot_x: {rotation[1]:.2f} deg"
        )
        ax4.text(0.5, 0.5, txt, ha="center", va="center")
        ax4.axis("off")

        # OBSERVATION
        clusters = TissueClusters.from_labelmap_slice(self.cur_state_action.labels_2d_slice)
        show_clusters(clusters, sliced_img, ax5)

        # REWARD
        ax5.text(0, 0, f"Reward: {self.cur_reward:.2f}", fontsize=12, color="red")

        plt.close()
        return fig

    def _create_figure_axis(self) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(9, 6))
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
