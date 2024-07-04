import numpy as np
import SimpleITK as sitk
from armscan_env.config import get_config

config = get_config()


def resize_sitk_volume(
    volumes: list[sitk.Image],  # n_spacing: tuple[float, float, float],
) -> list[sitk.Image]:
    """Resize a SimpleITK volume to a normalized spacing, and interpolate to get right amount of voxels.
    Have a look at [this](https://stackoverflow.com/questions/48065117/simpleitk-resize-images) link to see potential problems.

    :param volumes: the volumes to resize
    :param n_spacing: the normalized spacing to set
    :return: the resized volume
    """
    volumes[0].GetDimension()

    # Reference spacing will be the smallest spacing in the dataset
    reference_spacing = np.min([volume.GetSpacing() for volume in volumes], axis=0)

    normalized_volumes = []
    for _i, volume in enumerate(volumes):
        volume_physical_size = [
            sz * spc for sz, spc in zip(volume.GetSize(), volume.GetSpacing(), strict=True)
        ]
        # Size will be adjusted based on the new volume spacing
        reference_size = [
            int(phys_sz / spc)
            for phys_sz, spc in zip(volume_physical_size, reference_spacing, strict=True)
        ]

        # Resample the image to the reference image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(volume)
        resampler.SetOutputSpacing(reference_spacing)
        resampler.SetSize(reference_size)
        resampler.SetOutputDirection(volume.GetDirection())
        resampler.SetOutputOrigin(volume.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        normalized_volumes.append(resampler.Execute(volume))

    return normalized_volumes


def load_sitk_volumes(
    normalize: bool = False,
) -> list[sitk.Image]:
    """Load a SimpleITK volume from a file.

    :param normalize: whether to normalize the volumes to a single spacing
    :return: the loaded volume
    """
    volumes = []
    # count how many nii files are under the path and load them with config.get_labels_patt
    for label in range(1, config.count_labels() + 1):
        volume = sitk.ReadImage(config.get_labels_path(label))
        volumes.append(volume)

    if normalize:
        volumes = resize_sitk_volume(volumes)

    return volumes
