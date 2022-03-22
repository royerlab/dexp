import click
from arbol import asection

from dexp.cli.defaults import DEFAULT_CLEVEL, DEFAULT_CODEC, DEFAULT_STORE
from dexp.cli.parsing import (
    _get_output_path,
    _parse_channels,
    _parse_chunks,
    _parse_slicing,
)
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.denoise import dataset_denoise
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@click.argument("input_paths", nargs=-1)  # ,  help='input path'
@click.option("--output_path", "-o")  # , help='output path'
@click.option("--channels", "-c", default=None, help="list of channels, all channels when ommited.")
@click.option(
    "--slicing",
    "-s",
    default=None,
    help="dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ",
)
@click.option("--tilesize", "-ts", type=int, default=320, help="Tile size for tiled computation", show_default=True)
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
def denoise(
    input_paths,
    output_path,
    channels,
    slicing,
    tilesize,
    store,
    chunks,
    codec,
    clevel,
    overwrite,
):
    """TODO"""

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, channels)
    slicing = _parse_slicing(slicing)

    if slicing is not None:
        input_dataset.set_slicing(slicing)

    output_path = _get_output_path(input_paths[0], output_path, "_denoised")
    mode = "w" + ("" if overwrite else "-")
    output_dataset = ZDataset(
        output_path,
        mode=mode,
        store=store,
        codec=codec,
        clevel=clevel,
        chunks=_parse_chunks(chunks),
        parent=input_dataset,
    )

    with asection(f"Denoising: {input_paths} to {output_path} for channels {channels}"):
        dataset_denoise(
            input_dataset,
            output_dataset,
            channels=channels,
            tilesize=tilesize,
        )

    input_dataset.close()
    output_dataset.close()
