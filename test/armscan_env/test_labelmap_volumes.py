import os

import numpy as np
import pytest
import SimpleITK as sitk
from armscan_env.clustering import TissueLabel
from armscan_env.config import get_config
from armscan_env.volumes.slicing import slice_volume

config = get_config()


@pytest.fixture(scope="session")
def labelmaps():
    result = [
        (sitk.ReadImage(config.get_labels_path(i), i), i)
        for i in range(1, len(os.listdir(config.get_labels_basedir())))
    ]
    if not result:
        raise ValueError("No labelmaps files found in the labels directory")
    return result


class TestLabelMaps:
    @staticmethod
    def test_no_empty_labelmaps(labelmaps):
        for labelmap, _i in labelmaps:
            assert labelmap.GetSize() != (0, 0, 0)

    @staticmethod
    def test_all_tissue_labels_present(labelmaps):
        for labelmap, _i in labelmaps:
            img_array = sitk.GetArrayFromImage(labelmap)
            for label in TissueLabel:
                assert np.any(img_array == label.value)

    @staticmethod
    def test_labelmap_properly_sliced(labelmaps):
        for labelmap, _i in labelmaps:
            slice_shape = (labelmap.GetSize()[0], labelmap.GetSize()[2])
            sliced_volume = slice_volume(
                volume=labelmap,
                slice_shape=slice_shape,
                y_trans=-labelmap.GetOrigin()[1],
            )
            sliced_img = sitk.GetArrayFromImage(sliced_volume)[:, 0, :]
            assert not np.all(sliced_img == 0)
