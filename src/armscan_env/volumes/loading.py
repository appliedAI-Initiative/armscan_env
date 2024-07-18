from enum import Enum

import numpy as np
import SimpleITK as sitk
from armscan_env.config import get_config
from armscan_env.envs.state_action import ManipulatorAction
from armscan_env.volumes.volumes import ImageVolume

config = get_config()


class RegisteredLabelmap(Enum):
    v1 = 1
    v2 = 2
    v13 = 13
    v17 = 17
    v18 = 18
    v35 = 35
    v42 = 42

    def get_optimal_action(self) -> ManipulatorAction:
        match self:
            case RegisteredLabelmap.v1:
                return ManipulatorAction(rotation=(19.3, 0.0), translation=(0.0, 140.0))
            case RegisteredLabelmap.v2:
                return ManipulatorAction(rotation=(5, 0), translation=(0, 112))
            case RegisteredLabelmap.v13:
                return ManipulatorAction(rotation=(5, 0), translation=(0, 165))
            case RegisteredLabelmap.v17:
                return ManipulatorAction(rotation=(5, 0), translation=(0, 158))
            case RegisteredLabelmap.v18:
                return ManipulatorAction(rotation=(0, 0), translation=(0, 105))
            case RegisteredLabelmap.v35:
                return ManipulatorAction(rotation=(3, 0), translation=(0, 155))
            case RegisteredLabelmap.v42:
                return ManipulatorAction(rotation=(-3, 0), translation=(0, 178))
            case _:
                raise ValueError(f"Optimal action for {self} not defined")

    def get_labelmap_id(self) -> int:
        return self.value

    def get_file_path(self) -> str:
        return config.get_single_labelmap_path(self.get_labelmap_id())

    def load_labelmap(self) -> ImageVolume:
        volume = sitk.ReadImage(self.get_file_path())
        optimal_action = self.get_optimal_action()
        return ImageVolume(volume, optimal_action=optimal_action)

    @classmethod
    def load_all_labelmaps(cls, normalize_spacing: bool = True) -> list[ImageVolume]:
        volumes = [labelmap.load_labelmap() for labelmap in cls]
        if normalize_spacing:
            volumes = normalize_sitk_volumes_to_highest_spacing(volumes)
        return volumes


def normalize_sitk_volumes_to_highest_spacing(
    volumes: list[ImageVolume],  # n_spacing: tuple[float, float, float],
) -> list[ImageVolume]:
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
        normalized_volume = ImageVolume(
            resampler.Execute(volume),
            optimal_action=volume.optimal_action,
        )
        normalized_volumes.append(normalized_volume)

    return normalized_volumes


def load_sitk_volumes(
    normalize: bool = False,
) -> list[ImageVolume]:
    """Load a SimpleITK volume from a file.

    :param normalize: whether to normalize the volumes to a single spacing
    :return: the loaded volume
    """
    return RegisteredLabelmap.load_all_labelmaps(normalize_spacing=normalize)
