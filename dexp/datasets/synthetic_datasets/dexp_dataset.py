from dexp.datasets.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.datasets.synthetic_datasets.nuclei_background_data import (
    generate_nuclei_background_data,
)
from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.deskew import skew
from dexp.utils.backends import BestBackend


def generate_dexp_zarr_dataset(path: str, dataset_type: str, n_time_pts: int = 2, **kwargs) -> ZDataset:
    """Auxiliary function to generate dexps' zarr dataset from synthetic datas.

    Parameters
    ----------
    path : str
        Dataset path.

    dataset_type : str
        Dataset type, options are `fusion` or `nuclei`.

    n_time_pts : int, optional
        Number of time points (repeats), by default 2

    Returns
    -------
    ZDataset
        Generated dataset object
    """
    DS_TYPES = ("fusion", "nuclei", "nuclei-skewed")
    if dataset_type not in DS_TYPES:
        raise ValueError(f"`dataset_type` must be {DS_TYPES}, found {dataset_type}")

    ds = ZDataset(path, mode="w")

    if dataset_type == "fusion":
        names = ["ground-truth", "low-quality", "blend-a", "blend-b", "image-C0L0", "image-C1L0"]
        images = generate_fusion_test_data(**kwargs)

    elif dataset_type.startswith("nuclei"):
        names = ["ground-truth", "background", "image"]
        images = generate_nuclei_background_data(**kwargs)

        if dataset_type.endswith("skewed"):
            # skewing only the `image` channel
            with BestBackend() as bkd:
                xp = bkd.get_xp_module()
                skewed = bkd.to_numpy(skew(bkd.to_backend(images[-1]), xp.ones((5,) * images[-1].ndim), 1, 45, 1))
                images = images[:-1] + (skewed,)
    else:
        raise NotImplementedError

    for name, image in zip(names, images):
        ds.add_channel(name, (n_time_pts,) + image.shape, dtype=image.dtype)
        for t in range(n_time_pts):
            # adding a small shift to each time point
            ds.write_stack(name, t, image)

    ds.check_integrity()

    return ds
