# my custom loss function for image navigation
import logging
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from armscan_env.clustering import TissueClusters
from armscan_env.envs.base import RewardMetric
from armscan_env.envs.state_action import LabelmapStateAction

log = logging.getLogger(__name__)


@lru_cache(maxsize=100)
def anatomy_based_rwd(
    tissue_clusters: TissueClusters,
    n_landmarks: tuple[int, int, int] = (4, 3, 1),
) -> float:
    """Calculate the reward based on the presence and location of anatomical landmarks.

    :param tissue_clusters: dictionary of tissues and their clusters
    :param n_landmarks: number of landmarks for each tissue,
        in the same order as TissueLabel
    :return: reward value.
    """
    bones_loss = abs(len(tissue_clusters.bones) - n_landmarks[0])
    ligament_loss = abs(len(tissue_clusters.tendons) - n_landmarks[1])
    ulnar_loss = abs(len(tissue_clusters.ulnar) - n_landmarks[2])

    landmark_loss = bones_loss + ligament_loss + ulnar_loss
    log.debug(f"{bones_loss=} +\n{ligament_loss=} +\n{ulnar_loss=} =\n{landmark_loss=}")

    missing_landmark_loss = 0  # Absence of landmarks
    location_loss = 1  # Location of landmarks

    if len(tissue_clusters.bones) == 0:
        missing_landmark_loss += 1
        log.debug("No bones found")
    if len(tissue_clusters.tendons) == 0:
        missing_landmark_loss += 1
        log.debug("No tendons found")
    if len(tissue_clusters.ulnar) == 0:
        missing_landmark_loss += 1
        log.debug("No ulnar artery found")

    if (
        len(tissue_clusters.bones) != 0
        and len(tissue_clusters.tendons) != 0
        and len(tissue_clusters.ulnar) == 1
    ):
        bones_centers = np.array([cluster.center for cluster in tissue_clusters.bones])
        bones_centers_mean = np.mean(bones_centers, axis=0)
        log.debug(f"{bones_centers_mean=}")

        tendons_centers = np.array([cluster.center for cluster in tissue_clusters.tendons])
        tendons_centers_mean = np.mean(tendons_centers, axis=0)
        log.debug(f"{tendons_centers_mean=}")

        # There must be only one ulnar tissue so there is no need to take the mean
        ulnar_center = tissue_clusters.ulnar[0].center
        log.debug(f"{ulnar_center=}")

        # Check the orientation of the arm:
        # The bones center might be over or under the tendons center depending on the origin
        orientation = (bones_centers_mean[1] - tendons_centers_mean[1]) // abs(
            bones_centers_mean[1] - tendons_centers_mean[1],
        )
        log.debug(f"{orientation=}")

        # Ulnar artery must be under tendons in the positive orientation:
        if orientation * ulnar_center[1] < orientation * tendons_centers_mean[1]:
            location_loss = 0
        else:
            log.debug("Ulnar center not where expected")

    # Loss is bounded between 0 and 1
    loss = (1 / 3) * (
        (1 / sum(n_landmarks)) * landmark_loss + (1 / 3) * missing_landmark_loss + location_loss
    )

    log.debug(
        f"Landmark loss: {landmark_loss}\n"
        f"Missing landmark loss: {missing_landmark_loss}\n"
        f"Location loss: {location_loss}\n"
        f"Total loss: {loss}",
    )

    return -loss


class LabelmapClusteringBasedReward(RewardMetric[LabelmapStateAction]):
    """Reward metric based on the presence and location of anatomical landmarks.
    The reward is calculated as the negative of the loss function calculated by anatomy_based_rwd.
    """

    def __init__(
        self,
        n_landmarks: Sequence[int] = (4, 3, 1),
    ):
        self.n_landmarks = n_landmarks

    def compute_reward(self, state: LabelmapStateAction) -> float:
        clusters = TissueClusters.from_labelmap_slice(state.labels_2d_slice)
        return anatomy_based_rwd(tissue_clusters=clusters, n_landmarks=self.n_landmarks)

    @property
    def range(self) -> tuple[float, float]:
        """Reward range is [-1, 0], where 0 is the best reward and -1 is the worst."""
        return -1.0, 0.0
