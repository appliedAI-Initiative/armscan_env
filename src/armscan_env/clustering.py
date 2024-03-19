import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DataCluster:
    cluster: list[tuple[float, float]] | np.ndarray
    center: tuple[np.floating[Any], np.floating[Any]]


@dataclass(kw_only=True)
class TissueClusters:
    bones: list[DataCluster]
    tendons: list[DataCluster]
    ulnar: list[DataCluster]

    def __iter__(self) -> Iterator[list[DataCluster]]:
        return iter([self.bones, self.tendons, self.ulnar])


def find_clusters(tissue_value: int, slice: np.ndarray) -> list[DataCluster]:
    """Find clusters of a given tissue in a slice
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
                cluster=list(zip(cluster_indices[0], cluster_indices[1], strict=True)),
                center=center,
            ),
        )
    return cluster_list


def cluster_iter(tissues: dict, slice: np.ndarray) -> TissueClusters:
    """Find clusters of all tissues in a slice
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
    tissue_value: int,
    slice: np.ndarray,
    eps: float,
    min_samples: int,
) -> list[DataCluster]:
    """Find clusters of a given tissue in a slice using DBSCAN
    :param tissue_value: value of the tissue to cluster
    :param slice: image slice to cluster
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: list of clusters and their centers.
    """
    binary_mask = slice == tissue_value
    if np.all(binary_mask == 0):
        log.debug("No tissues to cluster with given value.")
        return []

    label_positions = np.array(list(zip(*np.where(binary_mask), strict=True)))
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clusterer.fit_predict(label_positions)
    n_clusters = (
        len(np.unique(clusters)) - 1
    )  # noise cluster has label -1, we don't take it into account
    log.debug(f"Found {n_clusters} clusters")

    cluster_list = []
    for cluster in range(n_clusters):
        label_to_pos_array = label_positions[clusters == cluster]  # get positions of each cluster
        cluster_centers = np.mean(label_to_pos_array, axis=0)  # mean of each column

        cluster_list.append(DataCluster(cluster=label_to_pos_array, center=cluster_centers))

    return cluster_list


# TODO: set different parameters for each tissue from config
def DBSCAN_cluster_iter(tissues: dict, slice: np.ndarray) -> TissueClusters:
    """Find clusters of all tissues in a slice using DBSCAN
    :param tissues: dictionary of tissues and their values
    :param slice: image slice to cluster
    :return: dictionary of tissues and their clusters.
    """
    bones = find_DBSCAN_clusters(tissues["bones"], slice, eps=4.1, min_samples=46)
    tendons = find_DBSCAN_clusters(tissues["tendons"], slice, eps=4.1, min_samples=46)
    ulnar = find_DBSCAN_clusters(tissues["ulnar"], slice, eps=2.5, min_samples=18)

    return TissueClusters(bones=bones, tendons=tendons, ulnar=ulnar)
