import os

from accsr.config import ConfigProviderBase, DefaultDataConfiguration

file_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

top_level_directory: str = os.path.abspath(os.path.join(file_dir, os.pardir, os.pardir))


class __Configuration(DefaultDataConfiguration):
    def get_labels_path(self, labelmap_number: int) -> str:
        labelmap_path = os.path.join(self.get_labels_basedir(), f"{labelmap_number:05d}_labels.nii")
        return self._adjusted_path(labelmap_path, relative=False, check_existence=True)

    def get_labels_basedir(self) -> str:
        return self._adjusted_path(
            os.path.join(self.data, "labels"),
            relative=False,
            check_existence=True,
        )

    def count_labels(self) -> int:
        labels_dir = self.get_labels_basedir()
        return len(
            [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))],
        )

    def get_mri_path(self, mri_number: int) -> str:
        mri_path = os.path.join(self.get_mri_basedir(), f"{mri_number:05d}.nii")
        return self._adjusted_path(mri_path, relative=False, check_existence=True)

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
