import glob
from functools import reduce
from os.path import exists, join
from typing import Sequence

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.joined_dataset import JoinedDataset
from dexp.datasets.tiff_dataset import TIFDataset
from dexp.datasets.zarr_dataset import ZDataset


def glob_datasets(glob_paths: Sequence[str]):
    """
    Opens a joined dataset given a list of path patterns (each following the 'glob' convention).

    Parameters
    ----------
    path: path of dataset

    Returns
    -------

    """
    if len(glob_paths) == 0:
        raise ValueError("No dataset path provided.")

    # Apply glob:
    paths = tuple(glob.glob(glob_path) for glob_path in glob_paths)

    if len(paths) == 0:
        raise ValueError("Could not find any dataset with provided paths", glob_paths)

    # concatenate list of paths:
    paths = reduce(lambda u, v: u + v, paths)

    # remove empty paths:
    paths = (path.strip() for path in paths)

    # remove empty paths:
    paths = (path for path in paths if len(path) > 0)

    # sort paths:
    paths = sorted(list(paths))

    return open_joined_datasets(paths), paths


def open_joined_datasets(paths: Sequence[str]):
    """
    Opens a joined dataset given a list of paths

    Parameters
    ----------
    path: path of dataset

    Returns
    -------
    JoinedDataset
    """

    if len(paths) == 1:
        # If there is only one dataset, no need to join anything:
        return open_dataset(paths[0])
    else:
        # If there are multiple datasets, we join them into a single dataset:
        datasets = tuple(open_dataset(path) for path in paths)
        dataset = JoinedDataset(datasets)
        return dataset


def open_dataset(path: str):
    """
    Opens a Zarr or ClearControl dataset given a path

    Parameters
    ----------
    path: path of dataset

    Returns
    -------
    dataset

    """

    # remove trailing slash:
    if path.endswith("/"):
        path = path[:-1]

    if path.endswith(".zarr.zip") or path.endswith(".zarr"):
        # we can recognise a Zarr dataset by its extension.
        dataset = ZDataset(path)
    elif path.endswith("tif") or path.endswith("tiff"):
        dataset = TIFDataset(path)
    elif exists(join(path, "stacks")):
        # we can recognise a ClearControl dataset by the presence of a 'stacks' sub folder.
        dataset = CCDataset(path)
    else:
        raise ValueError("Dataset type not recognised, or path incorrect!")

    return dataset
