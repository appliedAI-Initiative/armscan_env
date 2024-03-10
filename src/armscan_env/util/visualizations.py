import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def _show(
    slices: list,
    start: int,
    lap: int,
    col: int = 5,
    cmap: str | None = None,
    aspect: int = 6,
) -> AxesImage | Axes:
    """Function to display row of image slices
    :param slices: list of image slices
    :param start: starting slice number
    :param lap: number of slices to skip
    :param col: number of columns to display
    :param cmap: color map to use
    :param aspect: aspect ratio of each image
    :return: None.
    """
    rows = -(-len(slices) // col)
    fig, ax = plt.subplots(rows, col, figsize=(15, 2 * rows))
    # Flatten the ax array to simplify indexing
    ax = ax.flatten()
    for i, slice in enumerate(slices):
        ax[i].imshow(slice, cmap=cmap, origin="lower", aspect=aspect)
        ax[i].set_title(f"Slice {start - i * lap}")  # Set titles if desired
    # Adjust layout to prevent overlap of titles
    plt.tight_layout()
    return ax


def show_slices(
    data: np.ndarray,
    start: int,
    end: int,
    lap: int,
    col: int = 5,
    cmap: str | None = None,
    aspect: int = 6,
) -> AxesImage | Axes:
    """Function to display row of image slices
    :param data: 3D image data
    :param start: starting slice number
    :param end: ending slice number
    :param lap: number of slices to skip
    :param col: number of columns to display
    :param cmap: color map to use
    :param aspect: aspect ratio of each image
    :return: None.
    """
    it = 0
    slices = []
    for slice in range(start, 0, -lap):
        it += 1
        slices.append(data[:, slice, :])
        if it == end:
            break
    return _show(slices, start, lap, col, cmap, aspect)


def show_cluster_centers(
    tissue_clusters: dict,
    slice: np.ndarray,
    ax: Axes | None = None,
) -> AxesImage | Axes:
    """Plot the centers of the clusters of all tissues in a slice
    :param tissue_clusters: dictionary of tissues and their clusters
    :param slice: image slice to cluster
    :param ax: axis to plot on
    :return: None.
    """
    ax = ax or plt.gca()

    for tissue in tissue_clusters:
        for _label, data in enumerate(tissue_clusters[tissue]):
            # plot clusters with different colors
            ax.scatter(
                data["center"][1],
                data["center"][0],
                color="red",
                marker="*",
                s=20,
            )  # plot centers

    ax.imshow(slice, aspect=6, origin="lower")
    return ax


def show_clusters(
    tissue_clusters: dict,
    slice: np.ndarray,
    ax: Axes | None = None,
) -> AxesImage | Axes:
    """Plot the clusters of all tissues in a slice
    :param tissue_clusters: dictionary of tissues and their clusters
    :param slice: image slice to cluster
    :param ax: axis to plot on
    :return: None.
    """
    ax = ax or plt.gca()

    # create an empty array for cluster labels
    cluster_labels = slice.copy()

    for tissue in tissue_clusters:
        for label, data in enumerate(tissue_clusters[tissue]):
            # plot clusters with different colors
            cluster_labels[tuple(data["cluster"].T)] = (label + 1) * 10
            ax.scatter(data["center"][1], data["center"][0], color="red", marker="*", s=20)

    ax.imshow(cluster_labels, aspect=6, origin="lower")
    return ax


def show_only_clusters(
    tissue_clusters: dict,
    slice: np.ndarray,
    ax: Axes | None = None,
) -> AxesImage | Axes:
    """Plot only the clusters of all tissues in a slice
    :param tissue_clusters: dictionary of tissues and their clusters
    :param slice: image slice to cluster
    :param ax: axis to plot on
    :return: None.
    """
    ax = ax or plt.gca()

    # create an empty array for cluster labels
    cluster_labels = np.ones_like(slice) * 0

    for tissue in tissue_clusters:
        for label, data in enumerate(tissue_clusters[tissue]):
            # plot clusters with different colors
            cluster_labels[tuple(data["cluster"].T)] = (label + 1) * 10
            ax.scatter(
                data["center"][1],
                data["center"][0],
                color="red",
                marker="*",
                s=20,
            )  # plot centers
    ax.imshow(cluster_labels, aspect=6, origin="lower")
    return ax
