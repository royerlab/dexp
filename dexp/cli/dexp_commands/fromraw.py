import click
from arbol import asection

from dexp.cli.defaults import DEFAULT_CLEVEL, DEFAULT_CODEC, DEFAULT_STORE
from dexp.cli.parsing import _get_output_path, _parse_chunks
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.fromraw import dataset_fromraw
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@click.argument("input_paths", nargs=-1)  # ,  help='input path'
@click.option("--output_path", "-o")  # , help='output path'
@click.option("--ch-prefix", "-c", default=None, help="Prefix of channels, usually according to wavelength")
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
    "--workers",
    "-wk",
    default=-4,
    help="Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n| ",
    show_default=True,
)  #
def fromraw(
    input_paths,
    output_path,
    ch_prefix,
    store,
    chunks,
    codec,
    clevel,
    workers,
    overwrite,
):
    """Copies a dataset, channels can be selected, cropping can be performed, compression can be changed, ..."""

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, "_fromraw")
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

    with asection(f"Copying `fromraw`: {input_paths} to {output_path} for channels with prefix {ch_prefix}"):
        dataset_fromraw(
            input_dataset,
            output_dataset,
            ch_prefix,
            workers=workers,
        )

        input_dataset.close()
