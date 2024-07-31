import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


class TissueLabel(Enum):
    """Enum class for tissue labels in the labelmap volume."""

    BONES = 1
    TENDONS = 2
    ULNAR = 3

    def find_DBSCAN_clusters(self, labelmap_slice: np.ndarray) -> list["DataCluster"]:
        """Find clusters of a given tissue in a slice using DBSCAN."""
        match self:
            case TissueLabel.BONES:
                return find_DBSCAN_clusters(self, labelmap_slice, eps=4.1, min_samples=46)
            case TissueLabel.TENDONS:
                return find_DBSCAN_clusters(self, labelmap_slice, eps=3, min_samples=18)
            case TissueLabel.ULNAR:
                return find_DBSCAN_clusters(self, labelmap_slice, eps=1.1, min_samples=4)
            case _:
                raise ValueError(f"Unknown tissue label: {self}")


@dataclass(kw_only=True, frozen=True)
class DataCluster:
    """Data class for a cluster of a tissue in a slice."""

    datapoints: list[tuple[float, float]] | np.ndarray
    center: tuple[np.floating[Any], np.floating[Any]]

    def __hash__(self) -> int:
        return hash(tuple(self.datapoints) + self.center)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataCluster):
            return False
        return np.array_equal(self.datapoints, other.datapoints) and self.center == other.center


@dataclass(kw_only=True, frozen=True)
class TissueClusters:
    """Data class for all tissue clusters in a slice of a labelmap volume built using DBSCAN."""

    bones: list[DataCluster]
    tendons: list[DataCluster]
    ulnar: list[DataCluster]

    def __hash__(self) -> int:
        return hash(tuple(self.bones) + tuple(self.tendons) + tuple(self.ulnar))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TissueClusters):
            return False
        return (
            self.bones == other.bones
            and self.tendons == other.tendons
            and self.ulnar == other.ulnar
        )

    def get_cluster_for_label(self, label: TissueLabel) -> list[DataCluster]:
        """Get the clusters for a given tissue label."""
        match label:
            case TissueLabel.BONES:
                return self.bones
            case TissueLabel.TENDONS:
                return self.tendons
            case TissueLabel.ULNAR:
                return self.ulnar
            case _:
                raise ValueError(f"Unknown tissue label: {label}")

    @classmethod
    def from_labelmap_slice(cls, labelmap_slice: np.ndarray) -> Self:
        """Find clusters of all tissues in a slice using DBSCAN.

        :param labelmap_slice: image slice to cluster
        """
        bones_clusters = TissueLabel.BONES.find_DBSCAN_clusters(labelmap_slice)
        tendons_clusters = TissueLabel.TENDONS.find_DBSCAN_clusters(labelmap_slice)
        ulnar_clusters = TissueLabel.ULNAR.find_DBSCAN_clusters(labelmap_slice)

        return cls(bones=bones_clusters, tendons=tendons_clusters, ulnar=ulnar_clusters)


def find_clusters(tissue_value: int, slice: np.ndarray) -> list[DataCluster]:
    """Find clusters of a given tissue in a slice using a center-symmetric mask.

    :param tissue_value: value of the tissue to cluster
    :param slice: image slice to cluster
    :return: list of clusters and their centers.
    """
    binary_mask = slice == tissue_value
    if np.all(binary_mask is False):
        log.debug("No tissues to cluster. Please set values using set_values method.")
        return []

    labeled_array, num_clusters = label(binary_mask)
    cluster_list = []
    for cluster_label in range(num_clusters):
        cluster_indices = np.where(labeled_array == cluster_label + 1)
        center_x = np.mean(cluster_indices[0])
        center_y = np.mean(cluster_indices[1])
        center = (center_x, center_y)

        cluster_list.append(
            DataCluster(
                datapoints=list(zip(cluster_indices[0], cluster_indices[1], strict=True)),
                center=center,
            ),
        )
    return cluster_list


def cluster_iter(tissues: dict, slice: np.ndarray) -> TissueClusters:
    """Find clusters of all tissues in a slice using a center-symmetric mask.

    :param tissues: dictionary of tissues and their values
    :param slice: image slice to cluster
    :return: dictionary of tissues and their clusters.
    """
    tissues_clusters = TissueClusters(
        bones=find_clusters(tissues["bones"], slice),
        tendons=find_clusters(tissues["tendons"], slice),
        ulnar=find_clusters(tissues["ulnar"], slice),
    )
    log.debug(f"Found {len(tissues_clusters.bones)} bones clusters")
    log.debug(f"Found {len(tissues_clusters.tendons)} tendons clusters")
    log.debug(f"Found {len(tissues_clusters.ulnar)} ulnar clusters")

    return tissues_clusters


def find_DBSCAN_clusters(
    tissue_label: TissueLabel,
    labelmap_slice: np.ndarray,
    eps: float,
    min_samples: int,
) -> list[DataCluster]:
    """Find clusters of a given tissue in a slice using DBSCAN.

    :param label: value of the tissue to cluster
    :param labelmap_slice: slice of a labelmap volume, i.e., a 2D array with integers
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: list of clusters and their centers.
    """
    label_value = tissue_label.value
    binary_mask = labelmap_slice == label_value
    if np.all(binary_mask == 0):
        log.debug("No tissues to cluster with given value.")
        return []

    label_positions = np.array(list(zip(*np.where(binary_mask), strict=True)))
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clusterer.fit_predict(label_positions)
    if -1 in clusters:
        n_clusters = len(np.unique(clusters)) - 1
    else:
        n_clusters = len(np.unique(clusters))
    log.debug(f"Found {n_clusters} clusters")

    cluster_list = []
    for cluster in range(n_clusters):
        label_to_pos_array = label_positions[clusters == cluster]  # get positions of each cluster
        cluster_centers = np.mean(label_to_pos_array, axis=0)  # mean of each column

        cluster_list.append(DataCluster(datapoints=label_to_pos_array, center=cluster_centers))

    return cluster_list
