import click
from arbol import asection

from dexp.cli.defaults import DEFAULT_CLEVEL, DEFAULT_CODEC, DEFAULT_STORE
from dexp.cli.parsing import (
    _get_output_path,
    _parse_channels,
    _parse_chunks,
    _parse_devices,
    _parse_slicing,
)
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.segment import dataset_segment
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@click.argument("input_paths", nargs=-1)  # ,  help='input path'
@click.option("--output_path", "-o", default=None)  # , help='output path'
@click.option("--channels", "-c", default=None, help="list of channels, all channels when ommited.")
@click.option("--suffix", "-su", default="", help="suffix to append to results channels name.")
@click.option(
    "--slicing",
    "-s",
    default=None,
    help="dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ",
)
@click.option("--store", "-st", default=DEFAULT_STORE, help="Zarr store: ‘dir’, ‘ndir’, or ‘zip’", show_default=True)
@click.option("--chunks", "-chk", default=None, help="Dataset chunks dimensions, e.g. (1, 126, 512, 512).")
@click.option(
    "--codec",
    "-z",
    default=DEFAULT_CODEC,
    help="Compression codec: zstd for ’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ",
    show_default=True,
)
@click.option("--clevel", "-l", type=int, default=DEFAULT_CLEVEL, help="Compression level", show_default=True)
@click.option("--overwrite", "-w", is_flag=True, help="Forces overwrite of target", show_default=True)
@click.option(
    "--devices", "-d", type=str, default="0", help="Sets the CUDA devices id, e.g. 0,1,2 or ‘all’", show_default=True
)
@click.option("--z-scale", "-z", type=int, default=1, help="Anisotropy over z axis.")
@click.option(
    "--area-threshold",
    "-a",
    type=float,
    default=1e4,
    help="Parameter for cell detection, a smaller values will detect fewer segments.",
)
@click.option("--minimum-area", "-m", type=float, default=500, help="Minimum area (sum of pixels) per segment.")
@click.option(
    "--h-minima",
    "-h",
    type=float,
    default=1,
    help="Parameter to adjust the number of segments, smaller results in more segments.",
)
@click.option(
    "--compactness",
    "-cmp",
    type=float,
    default=0,
    help="Cell compactness (convexity) penalization, it penalizes non-convex shapes.",
)
@click.option(
    "--use-edt",
    "-e/-ne",
    is_flag=True,
    default=True,
    help="Use Euclidean Distance Transform (EDT) or image intensity for segmentation.",
)
def segment(
    input_paths,
    **kwargs,
):
    """Detects and segment cells using their intensity, returning a table with some properties"""

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, kwargs.pop("channels"))
    slicing = _parse_slicing(kwargs.pop("slicing"))
    devices = _parse_devices(kwargs.pop("devices"))

    if slicing is not None:
        input_dataset.set_slicing(slicing)

    output_path = _get_output_path(input_paths[0], kwargs.pop("output_path"), "_segments")
    mode = "w" + ("" if kwargs.pop("overwrite") else "-")
    output_dataset = ZDataset(
        output_path,
        mode=mode,
        store=kwargs.pop("store"),
        codec=kwargs.pop("codec"),
        clevel=kwargs.pop("clevel"),
        chunks=_parse_chunks(kwargs.pop("chunks")),
        parent=input_dataset,
    )

    with asection(f"Denoising: {input_paths} to {output_path} for channels {channels}"):
        dataset_segment(
            input_dataset,
            output_dataset,
            channels=channels,
            devices=devices,
            **kwargs,
        )

    input_dataset.close()
    output_dataset.close()
