import numpy as np
import pytest
import SimpleITK as sitk
from armscan_env.clustering import TissueClusters, TissueLabel
from armscan_env.config import get_config
from armscan_env.envs.labelmaps_navigation import VOL_NAME_TO_OPTIMAL_ACTION
from armscan_env.envs.rewards import anatomy_based_rwd
from armscan_env.envs.state_action import ManipulatorAction
from armscan_env.volumes.loading import load_sitk_volumes
from armscan_env.volumes.slicing import get_volume_slice

config = get_config()


@pytest.fixture(scope="session")
def labelmaps():
    result = load_sitk_volumes(normalize=False)
    if not result:
        raise ValueError("No labelmaps files found in the labels directory")
    return result


class TestLabelMaps:
    @staticmethod
    def test_no_empty_labelmaps(labelmaps):
        for labelmap in labelmaps:
            assert labelmap.GetSize() != (0, 0, 0)

    @staticmethod
    def test_all_tissue_labels_present(labelmaps):
        for labelmap in labelmaps:
            img_array = sitk.GetArrayFromImage(labelmap)
            for label in TissueLabel:
                assert np.any(img_array == label.value)

    @staticmethod
    def test_labelmap_properly_sliced(labelmaps):
        for labelmap in labelmaps:
            slice_shape = (labelmap.GetSize()[0], labelmap.GetSize()[2])
            sliced_volume = get_volume_slice(
                volume=labelmap,
                slice_shape=slice_shape,
                action=ManipulatorAction(
                    rotation=(0.0, 0.0),
                    translation=(0.0, -labelmap.GetOrigin()[1]),
                ),
            )
            sliced_img = sitk.GetArrayFromImage(sliced_volume)[:, 0, :]
            assert not np.all(sliced_img == 0)

    @staticmethod
    def test_optimal_actions(labelmaps):
        for i, labelmap in enumerate(labelmaps):
            optimal_action = VOL_NAME_TO_OPTIMAL_ACTION[str(i + 1)]
            slice_shape = (labelmap.GetSize()[0], labelmap.GetSize()[2])
            sliced_volume = get_volume_slice(
                volume=labelmap,
                slice_shape=slice_shape,
                action=optimal_action,
            )
            sliced_img = sitk.GetArrayFromImage(sliced_volume)[:, 0, :]
            cluster = TissueClusters.from_labelmap_slice(sliced_img.T)
            reward = anatomy_based_rwd(cluster)
            assert reward < 0.1
