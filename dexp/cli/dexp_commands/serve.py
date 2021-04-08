import click
from arbol.arbol import aprint, asection

from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.serve import dataset_serve


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--host', '-h', type=str, default="0.0.0.0", help='Host to serve from', show_default=True)
@click.option('--port', '-p', type=int, default=8000, help='Port to serve from', show_default=True)
def serve(input_paths,
          host,
          port):
    """ Serves dataset across network.
    """

    input_dataset, input_paths = glob_datasets(input_paths)

    with asection(f"Serving dataset(s): {input_paths}"):
        dataset_serve(input_dataset,
                      host=host,
                      port=port)

        input_dataset.close()
        aprint("Done!")
