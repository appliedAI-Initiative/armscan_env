from typing import Any

import numpy as np
from numpy import dtype, ndarray
from scipy.ndimage import label
from sklearn.cluster import DBSCAN


def find_clusters(tissue_value: int, slice: np.ndarray) -> list[dict]:
    """Find clusters of a given tissue in a slice
    :param tissue_value: value of the tissue to cluster
    :param slice: image slice to cluster
    :return: list of clusters and their centers.
    """
    # Create a binary mask based on the threshold
    binary_mask = slice == tissue_value

    # Check if there are tissues with given label
    if np.all(binary_mask is False):
        print("No tissues to cluster. Please set values using set_values method.")
        return []

    # Label connected components in the binary mask
    labeled_array, num_clusters = label(binary_mask)

    # Extract clusters and their centers
    cluster_data = []

    for cluster_label in range(num_clusters):
        cluster_indices = np.where(labeled_array == cluster_label + 1)
        # Calculate the center of the cluster
        center_x = np.mean(cluster_indices[0])
        center_y = np.mean(cluster_indices[1])
        center = (center_x, center_y)

        # Save both the cluster and center under the same key
        cluster_data.append(
            {
                "cluster": np.array(
                    list(zip(cluster_indices[0], cluster_indices[1], strict=False)),
                ),
                "center": center,
            },
        )

    return cluster_data


def cluster_iter(tissues: dict, slice: np.ndarray) -> dict:
    """Find clusters of all tissues in a slice
    :param tissues: dictionary of tissues and their values
    :param slice: image slice to cluster
    :return: dictionary of tissues and their clusters.
    """
    # store clusters of tissues in a dict
    tissues_clusters = {}

    for tissue in tissues:
        print(f"Finding {tissue} clusters, with value {tissues[tissue]}:")
        tissues_clusters[tissue] = find_clusters(tissues[tissue], slice)

        print(f"Found {len(tissues_clusters[tissue])} clusters\n")
    print("---------------------------------------\n")
    return tissues_clusters


def find_DBSCAN_clusters(
    tissue_value: int,
    slice: np.ndarray,
    eps: float,
    min_samples: int,
) -> list[Any] | list[dict[str, ndarray[Any, dtype[Any]] | Any]]:
    """Find clusters of a given tissue in a slice using DBSCAN
    :param tissue_value: value of the tissue to cluster
    :param slice: image slice to cluster
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: list of clusters and their centers.
    """
    # binary filter for the tissue_value
    binary_mask = slice == tissue_value

    # Check if there are tissues with given tissue_value
    if np.all(binary_mask == 0):
        print("No tissues to cluster with given value.")
        return []

    # find label positions, upon which clustering wil be defined
    label_positions = np.array(list(zip(*np.where(binary_mask), strict=True)))
    # define clusterer
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    # find cluster prediction
    clusters = clusterer.fit_predict(label_positions)
    n_clusters = (
        len(np.unique(clusters)) - 1
    )  # noise cluster has label -1, we don't take it into account
    print(f"Found {n_clusters} clusters")

    # Extract clusters and their centers
    cluster_data = []

    for cluster in range(n_clusters):
        label_to_pos_array = label_positions[clusters == cluster]  # get positions of each cluster
        cluster_centers = np.mean(label_to_pos_array, axis=0)  # mean of each column
        # Save both the cluster and center under the same key
        cluster_data.append({"cluster": label_to_pos_array, "center": cluster_centers})

    return cluster_data


# TODO: set different parameters for each tissue
def DBSCAN_cluster_iter(tissues: dict, slice: np.ndarray, eps: float, min_samples: int) -> dict:
    """Find clusters of all tissues in a slice using DBSCAN
    :param tissues: dictionary of tissues and their values
    :param slice: image slice to cluster
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: dictionary of tissues and their clusters.
    """
    # store clusters of tissues in a dict
    tissues_clusters = {}

    for tissue in tissues:
        print(f"Finding {tissue} clusters, with value {tissues[tissue]}:")
        # find clusters for each tissue
        tissues_clusters[tissue] = find_DBSCAN_clusters(tissues[tissue], slice, eps, min_samples)

        # print the identified clusters and their centers
        for index, data in enumerate(tissues_clusters[tissue]):
            print(f"Center of {tissue} cluster {index}: {data['center']}")
    print("---------------------------------------\n")
    return tissues_clusters
