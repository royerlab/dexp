from typing import Any, List, Sequence, Tuple

import numpy
from arbol.arbol import aprint
from dask.array import concatenate

from dexp.datasets.base_dataset import BaseDataset

# Configure multithreading for Dask:


class JoinedDataset(BaseDataset):
    def __init__(self, datasets: Sequence[BaseDataset]):
        """Instanciates a joined dataset.

        Parameters
        ----------
        datasets : sequence of datasets to join into one temporally concatenated dataset.

        Returns
        -------
        Joined dataset
        """

        super().__init__(dask_backed=False)

        self._dataset_list: List[BaseDataset] = list(datasets)

        aprint(f"dataset list: {self._dataset_list}")

        # First we make sure that the list is not empty:
        if len(self._dataset_list) == 0:
            raise ValueError("Dataset list is empty!")

        # Second we check if the same channels are present in all datasets:
        _dataset_zero = self._dataset_list[0]
        for i, dataset in enumerate(self._dataset_list):
            if set(dataset.channels()) != set(_dataset_zero.channels()):
                aprint(
                    f"dataset #{i} has channels '{dataset.channels()}' but datatset #0"
                    + "has channels: '{_dataset_zero.channels()}'"
                )
                raise ValueError("All datasets must have the same exact channels!")

            # Third, we also check per channel if the shape and dtypes are the same
            for channel in _dataset_zero.channels():
                if _dataset_zero.shape(channel)[1:] != dataset.shape(channel)[1:]:
                    aprint(
                        f"dataset #{i} channel {channel} has shape '{dataset.shape(channel)[1:]}' "
                        + f"but datatset #0 channel {channel} has shape: '{_dataset_zero.shape(channel)[1:]}'"
                    )
                    raise ValueError(
                        "All datasets must have the same exact shape for the same channels!"
                        + " (except for time dimension!)"
                    )
                if _dataset_zero.dtype(channel) != dataset.dtype(channel):
                    aprint(
                        f"dataset #{i} channel {channel} has dtype '{dataset.dtype(channel)}' but datatset "
                        + f"#0 channel {channel} has dtype: '{_dataset_zero.dtype(channel)}'"
                    )
                    raise ValueError("All datasets must have the same exact dtype for the same channels!")

        # Build concatenated dask arrays:
        self._arrays = {}
        self._projection_arrays = {}
        for channel in self.channels():
            self._arrays[channel] = concatenate(
                dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for dataset in self._dataset_list
            )

            for axis in range(len(self.shape(channel)) - 1):
                try:
                    self._projection_arrays[f"{channel}/{axis}"] = concatenate(
                        dataset.get_projection_array(channel, axis=axis, wrap_with_dask=True)
                        for dataset in self._dataset_list
                    )
                except KeyError:
                    self._projection_arrays[f"{channel}/{axis}"] = None

    def close(self):
        for dataset in self._dataset_list:
            dataset.close()

    def check_integrity(self, channels: Sequence[str] = None) -> bool:
        for dataset in self._dataset_list:
            if not dataset.check_integrity(channels):
                return False
        return True

    def channels(self) -> Sequence[str]:
        return self._dataset_list[0].channels()

    def shape(self, channel: str) -> Sequence[int]:
        return (self.nb_timepoints(channel),) + tuple(self._dataset_list[0].shape(channel)[1:])

    def nb_timepoints(self, channel: str) -> int:
        return sum(dataset.nb_timepoints(channel) for dataset in self._dataset_list)

    def dtype(self, channel: str):
        return self._dataset_list[0].dtype(channel)

    def info(self, channel: str = None) -> str:
        if channel is not None:
            info_str = (
                f"Channel: '{channel}', nb time points: {self.shape(channel)[0]}, "
                + f"shape: {self.shape(channel)[1:]}, joined from {len(self._dataset_list)} datasets."
            )
            info_str += "\n"
            return info_str
        else:
            info_str = f"Joined dataset of length: {len(self._dataset_list)} \n"
            info_str += "\n\n"
            info_str += "Channels: \n"
            for channel in self.channels():
                info_str += "  └──" + self.info(channel) + "\n\n"
                for i, dataset in enumerate(self._dataset_list):
                    info_str += f"      dataset #{i}: nb time points: {dataset.nb_timepoints(channel)}\n"

            info_str += "\n\n"

            return info_str

    def get_metadata(self):
        """get the attributes stored in the zarr folder"""
        attrs = {}
        for dataset in self._dataset_list:
            attrs.update(dataset.get_metadata())
        return attrs

    def append_metadata(self, metadata: dict):
        raise NotImplementedError("Method append_metadata is not available for a joined dataset!")

    def get_array(self, channel: str, per_z_slice: bool = False, wrap_with_dask: bool = False):
        return self._arrays[channel]

    def get_stack(self, channel: str, time_point: int, per_z_slice: bool = False, wrap_with_dask: bool = False):
        return self._arrays[channel][time_point]

    def get_projection_array(self, channel: str, axis: int, wrap_with_dask: bool = False) -> Any:
        return self._projection_arrays[f"{channel}/{axis}"]

    def add_channel(self, name: str, shape: Tuple[int, ...], dtype, enable_projections: bool = True, **kwargs) -> Any:
        raise NotImplementedError("Cannot write to a joined dataset!")

    def write_stack(self, channel: str, time_point: int, stack_array: numpy.ndarray):
        raise NotImplementedError("Cannot write to a joined dataset!")

    def write_array(self, channel: str, array: numpy.ndarray):
        raise NotImplementedError("Cannot write to a joined dataset!")

    @property
    def path(self) -> str:
        return ",".join([ds.path for ds in self._dataset_list])
