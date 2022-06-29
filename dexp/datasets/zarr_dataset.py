import math as m
import os
import re
import shutil
import sys
from os.path import exists, isdir, isfile, join
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import dask
import numpy
import zarr
from arbol.arbol import aprint, asection
from ome_zarr.format import CurrentFormat
from zarr import Blosc, CopyError, convenience, open_group

from dexp.cli.defaults import DEFAULT_CLEVEL, DEFAULT_CODEC
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.ome_dataset import create_coord_transform, default_omero_metadata
from dexp.datasets.stack_iterator import StackIterator
from dexp.utils import compress_dictionary_lists_length
from dexp.utils.backends import Backend, BestBackend
from dexp.utils.config import config_blosc

try:
    from cucim import __version__ as sk_version
    from cucim.skimage.transform import downscale_local_mean

    DOWNSCALE_METHOD = "cucim.skimage.transform.downscale_local_mean"

except ImportError:
    from skimage import __version__ as sk_version
    from skimage.transform import downscale_local_mean

    DOWNSCALE_METHOD = "skimage.transform.downscale_local_mean"


class ZDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "r",
        store: str = None,
        *,
        codec: str = DEFAULT_CODEC,
        clevel: int = DEFAULT_CLEVEL,
        chunks: Optional[Sequence[int]] = None,
        parent: Optional[BaseDataset] = None,
    ):
        """Instantiates a Zarr dataset (and opens it)

        Parameters
        ----------
        path : path to zarr storage (directory or zip).
        mode : Access mode:
            'r' means read only (must exist);
            'r+' means read/write (must exist);
            'a' means read/write (create if doesn't exist);
            'w' means create (overwrite if exists);
            'w-' means create (fail if exists).
        store : type of store, can be 'dir', 'ndir', or 'zip'

        Returns
        -------
        Zarr dataset
        """
        config_blosc()

        super().__init__(dask_backed=False, path=path)

        self._store = None
        self._root_group = None

        self._codec = codec
        self._clevel = clevel
        self._chunks = chunks

        # Open remote store:
        if "http" in self._path:
            aprint(f"Opening a remote store at: {self._path}")
            from fsspec import get_mapper

            self.store = get_mapper(self._path)
            self._root_group = zarr.open(self.store, mode=mode)
            return

        # Correct path to adhere to convention:
        if "a" in mode or "w" in mode:
            if self._path.endswith(".zarr.zip") or store == "zip":
                self._path = self._path + ".zip" if self._path.endswith(".zarr") else self._path
                self._path = self._path if self._path.endswith(".zarr.zip") else self._path + ".zarr.zip"
            elif self._path.endswith(".nested.zarr") or self._path.endswith(".nested.zarr/") or store == "ndir":
                self._path = self._path if self._path.endswith(".nested.zarr") else self._path + ".nested.zarr"
            elif self._path.endswith(".zarr") or self._path.endswith(".zarr/") or store == "dir":
                self._path = self._path if self._path.endswith(".zarr") else self._path + ".zarr"

        # if exists and overwrite then delete!
        if exists(self._path) and mode == "w-":
            raise ValueError(f"Storage '{self._path}' already exists, add option '-w' to force overwrite!")
        elif exists(self._path) and mode == "w":
            aprint(f"Deleting '{self._path}' for overwrite!")
            if isdir(self._path):
                # This is a very dangerous operation, let's double check that the folder really holds a zarr dataset:
                _zgroup_file = join(self._path, ".zgroup")
                _zarray_file = join(self._path, ".zarray")
                # We check that either of these two hiddwn files are present, and if '.zarr' is part of the name:
                if (".zarr" in self._path) and (exists(_zgroup_file) or exists(_zarray_file)):
                    shutil.rmtree(self._path, ignore_errors=True)
                else:
                    raise ValueError(
                        "Specified path does not seem to be a zarr dataset, deletion for overwrite"
                        "not performed out of abundance of caution, check path!"
                    )

            elif isfile(self._path):
                os.remove(self._path)

        if exists(self._path):
            aprint(f"Opening existing Zarr: '{self._path}' with read/write mode: '{mode}' and store type: '{store}'")
            if isfile(self._path) and (self._path.endswith(".zarr.zip") or store == "zip"):
                aprint("Opening as ZIP store")
                self._store = zarr.storage.ZipStore(self._path)
            elif isdir(self._path) and (
                self._path.endswith(".nested.zarr") or self._path.endswith(".nested.zarr/") or store == "ndir"
            ):
                aprint("Opening as Nested Directory store")
                self._store = zarr.storage.NestedDirectoryStore(self._path)
            elif isdir(self._path) and (
                self._path.endswith(".zarr") or self._path.endswith(".zarr/") or store == "dir"
            ):
                aprint("Opening as Directory store")
                self._store = zarr.storage.DirectoryStore(self._path)

            aprint(f"Opening with mode: {mode}")
            self._root_group = open_group(self._store, mode=mode)

        elif "a" in mode or "w" in mode:
            aprint(f"Creating Zarr storage: '{self._path}' with read/write mode: '{mode}' and store type: '{store}'")
            if store is None:
                store = "dir"
            try:
                if self._path.endswith(".zarr.zip") or store == "zip":
                    aprint("Opening as ZIP store")
                    self._store = zarr.storage.ZipStore(self._path)
                elif self._path.endswith(".nested.zarr") or self._path.endswith(".nested.zarr/") or store == "ndir":
                    aprint("Opening as Nested Directory store")
                    self._store = zarr.storage.NestedDirectoryStore(self._path)
                elif self._path.endswith(".zarr") or self._path.endswith(".zarr/") or store == "dir":
                    aprint("Opening as Directory store")
                    self._store = zarr.storage.DirectoryStore(self._path)
                else:
                    aprint(
                        f"Cannot open {self._path}, needs to be a zarr directory (directory that ends with `.zarr` or "
                        + "`.nested.zarr` for nested folders), or a zipped zarr file (file that ends with `.zarr.zip`)"
                    )

                self._root_group = zarr.convenience.open(self._store, mode=mode)

            except Exception:
                raise ValueError(
                    "Problem: can't create target file/directory, most likely the target dataset "
                    + f"already exists or path incorrect: {self._path}"
                )
        else:
            raise ValueError(f"Invalid read/write mode or invalid path: {self._path} (check path!)")

        # updating metadata
        if parent is not None:
            metadata = parent.get_metadata()
            metadata.pop("cli_history", None)  # avoiding adding it twice
            self.append_metadata(metadata)

        if mode in ("a", "w", "w-"):
            self.append_cli_history(parent if isinstance(parent, ZDataset) else None)

    def chunk_shape(self, channel: str) -> Sequence[int]:
        return self.get_array(channel).chunks

    @staticmethod
    def _default_chunks(shape: Tuple[int], dtype: Union[str, numpy.dtype], max_size: int = 2147483647) -> Tuple[int]:
        if not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)
        width = shape[-1]
        height = shape[-2]
        depth = min(max_size // (dtype.itemsize * width * height), shape[-3])
        chunk = (1, depth, height, width)
        return chunk[-len(shape) :]

    def close(self):
        # We close the store if it exists, i.e. if we have been writing to the dataset
        if self._store is not None:
            try:
                self._store.close()
            except AttributeError:
                pass

    def check_integrity(self, channels: Sequence[str] = None) -> bool:
        aprint("Checking integrity of zarr storage, might take some time.")
        if channels is None:
            channels = self.channels()
        for channel in channels:
            aprint(f"Checking integrity of channel '{channel}'...")
            array = self.get_array(channel, wrap_with_dask=False)
            if array.nchunks_initialized < array.nchunks:
                aprint(f"WARNING! not all chunks initialised! (dtype={array.dtype})")
                return False
            else:
                aprint(f"Channel '{channel}' seems ok!")
                return True

    def channels(self) -> Sequence[str]:
        return list(self._root_group.keys())

    def nb_timepoints(self, channel: str) -> int:
        return self.get_array(channel).shape[0]

    def shape(self, channel: str) -> Sequence[int]:
        return self.get_array(channel).shape

    def dtype(self, channel: str):
        return self.get_array(channel).dtype

    def info(self, channel: str = None, cli_history: bool = True) -> str:
        info_str = ""
        if channel is not None:
            info_str += (
                f"Channel: '{channel}', nb time points: {self.shape(channel)[0]}, shape: {self.shape(channel)[1:]}"
            )
            info_str += ".\n"
            info_str += str(self.get_array(channel).info)
            return info_str
        else:
            info_str += f"Dataset at location: {self._path} \n"
            info_str += f"Channels: {self.channels()} \n"
            info_str += "Zarr tree: \n"
            info_str += str(self._root_group.tree())
            info_str += ".\n\n"
            info_str += "Arrays: \n"
            for name in self.channels():
                info_str += "  │ \n"
                info_str += "  └──" + name + ":\n" + str(self.get_array(name).info) + "\n\n"
                info_str += ".\n\n"

        info_str += ".\n\n"
        info_str += "\nMetadata: \n"
        metadata = compress_dictionary_lists_length(self.get_metadata(), 5)
        for key, value in metadata.items():
            if "cli_history" not in key:
                info_str += f"\t{key} : {value} \n"

        if cli_history:
            info_str += ".\n\n"
            key = "cli_history"
            if key in self._root_group.attrs:
                info_str += "\nCommand line history:\n"
                commands_list = self._root_group.attrs[key]
                for command in commands_list[:-1]:
                    info_str += " ├──■ '" + command + "' \n"
                info_str += " └──■ '" + commands_list[-1] + "' \n"

        return info_str

    def get_metadata(self):
        """get the attributes stored in the zarr folder"""
        attrs = {}
        for name in self._root_group.attrs:
            attrs[name] = self._root_group.attrs[name]
        return attrs

    def append_metadata(self, metadata: dict):
        self._root_group.attrs.update(metadata)

    def append_cli_history(self, parent: Optional[BaseDataset]):
        key = "cli_history"
        cli_history = []
        if parent is not None:
            parent_metadata = parent.get_metadata()
            cli_history = parent_metadata.get(key, [])

        if key in self._root_group.attrs:
            cli_history += self._root_group.attrs[key]

        new_command = os.path.basename(sys.argv[0]) + " " + " ".join(sys.argv[1:])
        cli_history.append(new_command)
        self._root_group.attrs[key] = cli_history

    def _load_tensorstore(self, array: zarr.Array):
        import tensorstore as ts

        metadata = {
            "dtype": array.dtype.str,
            "order": array.order,
            "shape": array.shape,
        }
        ts_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(Path(self.path).resolve()),
            },
            "path": array.path,
            "metadata": metadata,
        }
        return ts.open(ts_spec, create=False, open=True).result()

    def get_array(
        self, channel: str, per_z_slice: bool = False, wrap_with_dask: bool = False, wrap_with_tensorstore: bool = False
    ) -> Union[zarr.Array, Any]:
        assert (wrap_with_dask != wrap_with_tensorstore) or not wrap_with_dask
        array = self._root_group[channel].get(channel)
        if wrap_with_dask:
            return dask.array.from_array(array, chunks=array.chunks)
        elif wrap_with_tensorstore:
            return self._load_tensorstore(array)
        return array

    def get_stack(self, channel: str, time_point: int, per_z_slice: bool = False, wrap_with_dask: bool = False):
        stack_array = self.get_array(channel, per_z_slice=per_z_slice, wrap_with_dask=wrap_with_dask)[time_point]
        return stack_array

    def get_projection_array(
        self, channel: str, axis: int, wrap_with_dask: bool = False, wrap_with_tensorstore: bool = False
    ) -> Optional[Union[zarr.Array, Any]]:
        assert (wrap_with_dask != wrap_with_tensorstore) or not wrap_with_dask
        array = self._root_group[channel].get(self._projection_name(channel, axis))
        if array is None:
            return None

        if wrap_with_dask:
            return dask.array.from_array(array, chunks=array.chunks)
        elif wrap_with_tensorstore:
            return self._load_tensorstore(array)
        return array

    def _projection_name(self, channel: str, axis: int):
        return f"{channel}_projection_{axis}"

    def write_stack(self, channel: str, time_point: int, stack_array: numpy.ndarray):
        array_in_zarr = self.get_array(channel=channel, wrap_with_dask=False)
        array_in_zarr[time_point] = stack_array

        for axis in range(stack_array.ndim):
            xp = Backend.get_xp_module()
            projection = xp.max(stack_array, axis=axis)
            projection_in_zarr = self.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
            projection_in_zarr[time_point] = projection

    def write_array(self, channel: str, array: numpy.ndarray):
        array_in_zarr = self.get_array(channel=channel, wrap_with_dask=False)
        array_in_zarr[...] = array

        for axis in range(array.ndim - 1):
            xp = Backend.get_xp_module()
            projection = xp.max(array, axis=axis + 1)
            projection_in_zarr = self.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
            projection_in_zarr[...] = projection

    def add_channel(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: numpy.dtype,
        chunks: Optional[Sequence[int]] = None,
        enable_projections: bool = True,
        codec: Optional[str] = None,
        clevel: Optional[int] = None,
        value: Optional[Any] = None,
    ) -> Any:
        """Adds a channel to this dataset

        Parameters
        ----------
        name : name of channel.
        shape : shape of correspodning array.
        dtype : dtype of array.
        chunks: chunks shape.
        codec: Compression codec to be used ('zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy').
        clevel: An integer between 0 and 9 specifying the compression level.

        Returns
        -------
        zarr array


        """
        # check if channel exists:
        if name in self.channels():
            raise ValueError("Channel already exist!")

        if chunks is None:
            if self._chunks is None:
                chunks = self._default_chunks(shape, dtype)
            else:
                chunks = self._chunks

        if clevel is None:
            clevel = self._clevel

        if codec is None:
            codec = self._codec

        aprint(f"chunks={chunks}")

        # Choosing the fill value to the largest value:
        fill_value = self._get_largest_dtype_value(dtype) if value is None else value

        aprint(
            f"Adding channel: '{name}' of shape: {shape}, chunks:{chunks}, dtype: {dtype}, "
            + f"fill_value: {fill_value}, codec: {codec}, clevel: {clevel}"
        )
        compressor = Blosc(cname=codec, clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        filters = []

        channel_group = self._root_group.create_group(name)
        array = channel_group.full(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            filters=filters,
            compressor=compressor,
            fill_value=fill_value,
        )

        if enable_projections:
            ndim = len(shape) - 1
            for axis in range(ndim):
                proj_name = self._projection_name(name, axis)

                proj_shape = list(shape)
                del proj_shape[1 + axis]
                proj_shape = tuple(proj_shape)

                # chunking along time must be 1 to allow parallelism, but no chunking for each projection (not needed!)
                proj_chunks = (1,) + (None,) * (len(chunks) - 2)

                channel_group.full(
                    name=proj_name,
                    shape=proj_shape,
                    dtype=dtype,
                    chunks=proj_chunks,
                    filters=filters,
                    compressor=compressor,
                    fill_value=fill_value,
                )

        return array

    def add_channels_to(
        self,
        zdataset: Union[str, "ZDataset"],
        channels: Sequence[str],
        rename: Sequence[str],
        store: str = None,
        add_projections: bool = True,
        overwrite: bool = True,
    ):
        """Adds channels from this zarr dataset into an other possibly existing zarr dataset

        Parameters
        ----------
        path : zarr dataset or path of zarr dataset.
        channels: list or tuple of channels to add
        rename: list or tuple of new names for channels
        store: type of zarr store: 'dir' or 'zip', only usefull if store does not exist yet!
        add_projections: If True the projections are also copied.
        overwrite: overwrite destination (not fully functional for zip stores!)

        """

        if type(zdataset) is str:
            zdataset = ZDataset(zdataset, "a", store, parent=self)

        root = zdataset._root_group

        aprint(f"Existing channels: {zdataset.channels()}")

        for channel, new_name in zip(channels, rename):
            try:
                array = self.get_array(channel, per_z_slice=False, wrap_with_dask=False)
                source_group = self._root_group[channel]
                source_arrays = source_group.items()

                aprint(f"Creating group for channel {channel} of new name {new_name}.")
                if new_name not in root.group_keys():
                    dest_group = root.create_group(new_name)
                else:
                    dest_group = root[new_name]

                aprint(
                    f"Fast copying channel {channel} renamed to {new_name} of shape {array.shape} and"
                    + f"dtype {array.dtype}"
                )

                for name, array in source_arrays:
                    if name in self.channels():
                        aprint(f"Fast copying array {name} to {new_name}")
                        convenience.copy(
                            source=array, dest=dest_group, name=new_name, if_exists="replace" if overwrite else "raise"
                        )

                        if add_projections:
                            ndim = array.ndim - 1
                            for axis in range(ndim):
                                proj_array = self.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
                                convenience.copy(
                                    source=proj_array,
                                    dest=dest_group,
                                    name=self._projection_name(new_name, axis),
                                    if_exists="replace" if overwrite else "raise",
                                )

            except (CopyError, NotImplementedError):
                aprint("Channel already exists, set option '-w' to force overwriting! ")

        zdataset.close()

    def get_resolution(self, channel: Optional[str] = None) -> List[float]:
        """
        Gets pixel resolution.

        Parameters
        ----------
        channel : Channel to obtain the information, if None returns the dataset default.
        """
        axes = ("dt", "dz", "dy", "dx")
        metadata = self.get_metadata()
        if channel is None:
            resolution = [metadata.get(axis, 1.0) for axis in axes]

        else:
            resolution = self.get_resolution()  # gets dataset default
            channel_metadata = metadata.get(channel, {})
            resolution = [channel_metadata.get(axis, s) for axis, s in zip(axes, resolution)]

        return resolution

    def get_translation(self, channel: Optional[str] = None) -> List[float]:
        """
        Gets channel translation.
        """
        axes = ("tt", "tz", "ty", "tx")
        metadata = self.get_metadata()
        if channel is None:
            translation = [metadata.get(axis, 0) for axis in axes]

        else:
            translation = self.get_translation()  # gets default translation
            channel_metadata = metadata.get(channel, {})
            translation = [channel_metadata.get(axis, t) for axis, t in zip(axes, translation)]

        return translation

    def to_bdv_format(self, channel: str, path: Union[str, Path]) -> None:
        import npy2bdv

        with asection("Writing BigDataViewer file"):
            if isinstance(path, str):
                path = Path(path)

            if path.exists():
                raise ValueError(f"Path: {path} exists!")

            array = self.get_array(channel)
            # ignoring time and reversing to fiji ordering
            resolution = self.get_resolution(channel)[1:][::-1]
            writer = npy2bdv.BdvWriter(str(path))

            for t in range(array.shape[0]):
                writer.append_view(array[t], time=t, calibration=resolution)
                aprint(f"Saved time point {t}")

            writer.write_xml()
            writer.close()

    def first_uninitialized_time_point(self, channel: str) -> int:
        """
        Returns the index of the first uninitialized time point or the last time point if it is fully initialized
        """
        array = self.get_array(channel)
        prog = re.compile(r"\.".join([r"\d+"] * min(1, array.ndim)))
        initialized = {
            int(k.split(".", 1)[0]) for k in zarr.storage.listdir(array.chunk_store, array._path) if prog.match(k)
        }
        return max(initialized)

    def to_ome_zarr(self, path: str, force_dtype: Optional[int] = None, n_scales: int = 3) -> None:
        ch = self.channels()[0]
        dexp_shape = self.shape(ch)

        dtype = force_dtype if force_dtype is not None else self.dtype(ch)

        for _ch in self.channels():
            if dexp_shape != self.shape(_ch):
                raise ValueError(
                    f"Channels {ch} and {_ch} have different "
                    f"shapes ({dexp_shape} and {self.shape(_ch)} "
                    "could not convert to ome-zarr."
                )
            if dtype != self.dtype(_ch) and force_dtype is None:
                raise ValueError(
                    f"Channels {ch} and {_ch} have different "
                    f"dtypes ({dtype} and {self.dtype(_ch)} "
                    "could not convert to ome-zarr."
                )

        ome_zarr_shape = (dexp_shape[0], len(self.channels()), *dexp_shape[1:])

        group = zarr.group(zarr.NestedDirectoryStore(path))

        arrays = []
        datasets = []
        for i in range(n_scales):
            factor = 2**i
            array_path = f"{i}"
            shape = ome_zarr_shape[:2] + tuple(int(m.ceil(s / factor)) for s in ome_zarr_shape[2:])
            chunks = (1,) + self._default_chunks(shape, dtype)
            ome_array = group.create_dataset(array_path, shape=shape, dtype=dtype, chunks=chunks)
            arrays.append(ome_array)
            datasets.append(
                {
                    "path": array_path,
                    "coordinateTransformations": create_coord_transform(self.get_resolution(), factor),
                }
            )

        with BestBackend() as bkd:
            for t in range(self.nb_timepoints(ch)):
                aprint(f"Converting time point {t} ...", end="\r")
                for c, channel in enumerate(self.channels()):
                    stack = self.get_stack(channel, t)
                    arrays[0][t, c] = stack
                    stack = bkd.to_backend(stack)
                    for i in range(1, n_scales):
                        factors = (2**i,) * stack.ndim
                        arrays[i][t, c] = bkd.to_numpy(downscale_local_mean(stack, factors))

        aprint("Done conversion to OME zarr")

        group.attrs["multiscales"] = [
            {
                "version": CurrentFormat().version,
                "datasets": datasets,
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "type": "local_mean",
                "metadata": {
                    "method": DOWNSCALE_METHOD,
                    "version": sk_version,
                },
            }
        ]
        group.attrs["omero"] = default_omero_metadata(self._path, self.channels(), dtype)

    def __getitem__(self, channel: str) -> StackIterator:
        return StackIterator(self.get_array(channel, wrap_with_tensorstore=True), self._slicing)

    def __contains__(self, channel: str) -> bool:
        """Checks if channels exists, valid for when using multiple process."""
        return channel in self._root_group
