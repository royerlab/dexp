from arbol.arbol import aprint

from dexp.datasets.zarr_dataset import ZDataset


def dataset_serve(dataset: ZDataset,
                  host: str = "0.0.0.0",
                  port: int = 8000):
    if not type(dataset) == ZDataset:
        aprint("Cannot serve a non-Zarr dataset!")
        return

    aprint(dataset.info())
    try:
        from simple_zarr_server import serve
        serve(dataset._root_group, host=host, port=port)
    finally:
        # close destination dataset:
        dataset.close()
