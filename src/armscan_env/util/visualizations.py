from typing import Any

import numpy as np
from armscan_env.clustering import TissueClusters, TissueLabel
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def _show(
    slices: list[np.ndarray],
    start: int,
    lap: int,
    col: int,
    extent: tuple[int, int, int, int] | None,
    cmap: str | None,
    axis: bool,
    **imshow_kwargs: Any,
) -> AxesImage | Axes:
    """Function to display row of image slices.

    :param slices: list of image slices
    :param start: starting slice number
    :param lap: number of slices to skip
    :param col: number of columns to display
    :param extent: extent of the image
    :param cmap: color map to use
    :param axis: whether to display axis
    :param imshow_kwargs: additional keyword arguments to pass to imshow
    :return: None.
    """
    if extent is None:
        if isinstance(slices[0], np.ndarray) and isinstance(slices[0].shape, tuple):
            extent = (0, slices[0].shape[0], 0, slices[0].shape[1])
        else:
            raise TypeError("Expected slice to be a numpy array with a shape attribute of type tuple.")

    rows = -(-len(slices) // col)
    fig, ax = plt.subplots(rows, col, figsize=(15, 2 * rows))
    # Flatten the ax array to simplify indexing
    ax = ax.flatten()
    for i, slice in enumerate(slices):
        ax[i].imshow(slice, cmap=cmap, origin="lower", extent=extent, **imshow_kwargs)
        ax[i].set_title(f"Slice {start - i * lap}")  # Set titles if desired
        ax[i].axis("off") if not axis else None  # Turn off axis if desired
    # Adjust layout to prevent overlap of titles
    plt.tight_layout()
    return ax


def show_slices(
    data: np.ndarray,
    start: int,
    end: int,
    lap: int,
    col: int = 5,
    extent: tuple[int, int, int, int] | None = None,
    cmap: str | None = None,
    axis: bool = False,
    **imshow_kwargs: Any,
) -> AxesImage | Axes:
    """Function to display row of image slices.

    :param data: 3D image data
    :param start: starting slice number
    :param end: ending slice number
    :param lap: number of slices to skip
    :param col: number of columns to display
    :param extent: extent of the image, can be set to change the reference frame
    :param cmap: color map to use
    :param axis: whether to display axis
    :param imshow_kwargs: additional keyword arguments to pass to imshow
    :return: None.
    """
    it = 0
    slices = []
    for slice in range(start, 0, -lap):
        it += 1
        slices.append(data[:, slice, :])
        if it == end:
            break
    return _show(slices, start, lap, col, extent, cmap, axis, **imshow_kwargs)


def show_clusters(
    tissue_clusters: TissueClusters,
    slice: np.ndarray,
    ax: Axes | None = None,
    extent: tuple[int, int, int, int] | None = None,
    **imshow_kwargs: Any,
) -> AxesImage | Axes:
    """Plot the clusters of all tissues in a slice.

    :param tissue_clusters: dictionary of tissues and their clusters
    :param slice: image slice to cluster
    :param ax: axis to plot on
    :param extent: extent of the image, can be set to change the reference frame
    :return: None.
    """
    ax = ax or plt.gca()

    if extent is None:
        if isinstance(slice, np.ndarray) and isinstance(slice.shape, tuple):
            extent = (0, slice.shape[0], 0, slice.shape[1])
        else:
            raise TypeError("Expected slice to be a numpy array with a shape attribute of type tuple.")

    cluster_labels = slice.copy()
    # Calculate the scaling factors based on the extent and slice shape
    x_scale = (extent[1] - extent[0]) / slice.shape[0]
    y_scale = (extent[3] - extent[2]) / slice.shape[1]

    for tissue in TissueLabel:
        for label, data in enumerate(tissue_clusters.get_cluster_for_label(tissue)):
            # plot clusters with different colors
            cluster_labels[tuple(np.array(data.datapoints).T)] = (label + 1) * 10
            x = data.center[0] * x_scale + extent[0]
            y = data.center[1] * y_scale + extent[2]
            # plot clusters with different colors
            ax.scatter(x, y, color="red", marker="*", s=20)
    ax.imshow(cluster_labels.T, origin="lower", extent=extent, **imshow_kwargs)
    return ax
