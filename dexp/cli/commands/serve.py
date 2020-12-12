import click
from arbol.arbol import aprint, asection

from dexp.cli.utils import _get_dataset_from_path
from dexp.datasets.operations.serve import dataset_serve


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--host', '-h', type=str, default="127.0.0.1", help='Host to serve from', show_default=True)
@click.option('--port', '-p', type=int, default=8000, help='Port to serve from', show_default=True)  #
def serve(input_path, host, port):
    input_dataset = _get_dataset_from_path(input_path)

    with asection(f"Serving dataset: {input_path}"):
        dataset_serve(input_dataset,
                      host=host,
                      port=port)

        input_dataset.close()
        aprint("Done!")
