import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import relabel_sequential


def roi_watershed_from_minima(image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """Computes the watershed from minima from pyift for each connected component ROI, it's useful to save memory.

    Parameters
    ----------
    image : np.ndarray
        Input gradient (basins) image.
    mask : np.ndarray
        Input mask for ROI computation.

    Returns
    -------
    np.ndarray
        Output watershed labels.
    """
    from pyift.shortestpath import watershed_from_minima

    labels = np.zeros(image.shape, dtype=np.int32)
    mask, _ = ndi.label(mask)

    offset = 1
    for slicing in ndi.find_objects(mask):
        if slicing is None:
            continue
        _, lb = watershed_from_minima(
            image[slicing],
            mask[slicing] > 0,
            **kwargs,
        )
        lb[lb < 0] = 0  # non-masked regions are -1
        lb, _, _ = relabel_sequential(lb, offset=offset)
        offset = lb.max() + 1
        labels[slicing] = lb

    return labels
