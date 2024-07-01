import SimpleITK as sitk


def load_sitk_volume(
    path: str,
    spacing: tuple[float, float, float] | None = (0.5, 0.5, 1),
) -> sitk.Image:
    """Load a SimpleITK volume from a file.

    :param path: path to the volume file
    :param spacing: spacing of the volume
    :return: the loaded volume
    """
    volume = sitk.ReadImage(path)
    if spacing is not None:
        volume.SetSpacing(spacing)
    return volume
