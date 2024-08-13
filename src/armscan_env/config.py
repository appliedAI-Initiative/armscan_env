import os

from accsr.config import ConfigProviderBase, DefaultDataConfiguration

file_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

top_level_directory: str = os.path.abspath(os.path.join(file_dir, os.pardir, os.pardir))


class __Configuration(DefaultDataConfiguration):
    def get_labelmap_file_ids(self) -> list[int]:
        labelmaps_dir = self.get_labelmaps_basedir()
        labels_numbers = sorted(
            [f[:5] for f in os.listdir(labelmaps_dir) if f.endswith("_labels.nii")],
        )
        return [int(f.lstrip("0")) for f in labels_numbers]

    def get_single_labelmap_path(self, labelmap_file_id: int) -> str:
        single_labelmap_path = os.path.join(
            self.get_labelmaps_basedir(),
            f"{labelmap_file_id:05d}_labels.nii",
        )
        return self._adjusted_path(single_labelmap_path, relative=False, check_existence=True)

    def get_labelmaps_path(self) -> list[str]:
        labelmaps_dir = self.get_labelmaps_basedir()
        labels_names = sorted([f for f in os.listdir(labelmaps_dir) if f.endswith("_labels.nii")])
        return [os.path.join(labelmaps_dir, labelmap_name) for labelmap_name in labels_names]

    def get_labelmaps_basedir(self) -> str:
        return self._adjusted_path(
            os.path.join(self.data, "labels"),
            relative=False,
            check_existence=True,
        )

    def get_cropped_labelmaps_basedir(self) -> str:
        return self._adjusted_path(
            os.path.join(self.data, "cropped"),
            relative=False,
            check_existence=True,
        )

    def get_single_cropped_labelmap_path(self, labelmap_file_id: int) -> str:
        single_labelmap_path = os.path.join(
            self.get_cropped_labelmaps_basedir(),
            f"{labelmap_file_id:05d}_cropped.nii",
        )
        return self._adjusted_path(single_labelmap_path, relative=False, check_existence=True)

    def count_labels(self) -> int:
        labels_dir = self.get_labelmaps_basedir()
        return len(
            [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))],
        )

    def get_single_mri_path(self, mri_number: int) -> str:
        single_mri_path = os.path.join(self.get_mri_basedir(), f"{mri_number:05d}.nii")
        return self._adjusted_path(single_mri_path, relative=False, check_existence=True)

    def get_mri_path(self) -> list[str]:
        mri_dir = self.get_mri_basedir()
        mri_names = sorted([f for f in os.listdir(mri_dir) if f.endswith(".nii")])
        return [os.path.join(mri_dir, labelmap_name) for labelmap_name in mri_names]

    def get_mri_basedir(self) -> str:
        return self._adjusted_path(
            os.path.join(self.data, "mri"),
            relative=False,
            check_existence=True,
        )

    def count_mri(self) -> int:
        mri_dir = self.get_mri_basedir()
        return len([f for f in os.listdir(mri_dir) if os.path.isfile(os.path.join(mri_dir, f))])


class ConfigProvider(ConfigProviderBase[__Configuration]):
    pass


_config_provider = ConfigProvider()


def get_config(reload: bool = False) -> __Configuration:
    """:param reload: if True, the configuration will be reloaded from the json files
    :return: the configuration instance
    """
    return _config_provider.get_config(reload=reload, config_directory=top_level_directory)
