import matplotlib.pyplot as plt
import numpy as np
import pytest
import SimpleITK as sitk
from armscan_env.clustering import TissueClusters, TissueLabel
from armscan_env.config import get_config
from armscan_env.envs.rewards import anatomy_based_rwd
from armscan_env.envs.state_action import ManipulatorAction
from armscan_env.util.visualizations import show_clusters
from armscan_env.volumes.loading import load_sitk_volumes
from armscan_env.volumes.volumes import TransformedVolume

config = get_config()


@pytest.fixture(scope="session")
def labelmaps():
    result = load_sitk_volumes(normalize=False)
    result.extend(load_sitk_volumes(normalize=True))
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
            sliced_volume = labelmap.get_volume_slice(
                slice_shape=slice_shape,
                action=ManipulatorAction(
                    rotation=(0.0, 0.0),
                    translation=(0.0, -labelmap.GetOrigin()[1]),
                ),
            )
            sliced_img = sitk.GetArrayFromImage(sliced_volume)
            assert not np.all(sliced_img == 0)

    @staticmethod
    def test_optimal_actions(labelmaps):
        for _i, labelmap in enumerate(labelmaps):
            slice_shape = (labelmap.GetSize()[0], labelmap.GetSize()[2])
            sliced_volume = labelmap.get_volume_slice(
                slice_shape=slice_shape,
                action=labelmap.optimal_action,
            )
            sliced_img = sitk.GetArrayFromImage(sliced_volume)
            cluster = TissueClusters.from_labelmap_slice(sliced_img.T)
            reward = anatomy_based_rwd(cluster)
            assert reward > -0.1

    @staticmethod
    def test_rand_transformations(labelmaps):
        for i, labelmap in enumerate(labelmaps):
            slice_shape = (labelmap.GetSize()[0], labelmap.GetSize()[2])
            j = 0
            while j < 3:
                volume_transformation_action = ManipulatorAction.sample()
                transformed_labelmap = TransformedVolume.create_transformed_volume(
                    volume=labelmap,
                    transformation_action=volume_transformation_action,
                )
                sliced_volume = transformed_labelmap.get_volume_slice(
                    slice_shape=slice_shape,
                    action=transformed_labelmap.optimal_action,
                )
                sliced_img = sitk.GetArrayFromImage(sliced_volume)
                cluster = TissueClusters.from_labelmap_slice(sliced_img.T)
                reward = anatomy_based_rwd(cluster)
                if reward < -0.1:
                    show_clusters(cluster, sliced_img.T)
                    print(
                        f"Volume {i + 1} and transformation {volume_transformation_action}, reward: {reward}",
                    )
                plt.show()
                j += 1
                assert (
                    reward > -0.1
                ), f"Reward: {reward} for volume {i + 1} and transformation {volume_transformation_action}"
