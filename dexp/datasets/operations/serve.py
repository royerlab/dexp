from arbol.arbol import aprint

from dexp.datasets.zarr_dataset import ZDataset


def dataset_serve(dataset: ZDataset, host: str = "127.0.0.1", port: int = 8000):

    if not type(dataset) == ZDataset:
        aprint("Cannot serve a non-Zarr dataset!")
        return

    aprint(dataset.info())
    try:
        from simple_zarr_server import serve
        # creates an in-memory store if not zarr.Array or zarr.Group
        serve(dataset._root_group, host=host, port=port)
    finally:
        dataset.close()
