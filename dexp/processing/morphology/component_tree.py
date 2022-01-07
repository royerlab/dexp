import higra as hg
import numpy as np
from scipy.signal.signaltools import _centered

from dexp.processing.morphology.utils import get_3d_image_graph
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def _generic_area_filtering(
    image: xpArray,
    area_threshold: float,
    tree_type: str,
    sampling: int,
) -> xpArray:
    """Applies an area filtering a down sampled version of the input given a component tree type.
    Note: Moves the data to the CPU.

    Parameters
    ----------
    image : xpArray
        Input grayscale image.
    area_threshold : float
        Threshold of area (number of pixels) after the sampling reduction.
    tree_type : str
        Type of tree for filtering computation, "max" ("min") used for opening (closing).
    sampling : int
        Down sampling rate for component tree operation.

    Returns
    -------
    xpArray
        Area filtered result.
    """

    if tree_type.lower() == "min":
        component_tree_fun = hg.component_tree_min_tree
    elif tree_type.lower() == "max":
        component_tree_fun = hg.component_tree_max_tree
    else:
        raise NotImplementedError

    sp = Backend.get_sp_module(image)

    small = sp.ndimage.zoom(image, zoom=1 / sampling, order=1)
    small = Backend.to_numpy(small)

    graph = get_3d_image_graph(small.shape)
    tree, alt = component_tree_fun(graph, small)

    area = hg.attribute_area(tree)
    filtered = hg.reconstruct_leaf_data(tree, alt, area < area_threshold)

    hg.clear_auto_cache()

    filtered = Backend.to_backend(filtered)

    return _centered(sp.ndimage.zoom(filtered, zoom=sampling, order=1), image.shape)


def area_opening(
    image: xpArray,
    area_threshold: float,
    sampling: int,
) -> xpArray:
    """Area opening on the downsampled input.
    Note: Moves the data to the CPU.

    Parameters
    ----------
    image : xpArray
        Input grayscale image.
    area_threshold : float
        Threshold of area (number of pixels) after the sampling reduction.
    sampling : int
        Down sampling rate for component tree operation.

    Returns
    -------
    xpArray
        Area opened result.
    """
    return _generic_area_filtering(image=image, area_threshold=area_threshold, tree_type="max", sampling=sampling)


def area_closing(
    image: xpArray,
    area_threshold: float,
    sampling: int,
) -> xpArray:
    """Area closing on the downsampled input.
    Note: Moves the data to the CPU.

    Parameters
    ----------
    image : xpArray
        Input grayscale image.
    area_threshold : float
        Threshold of area (number of pixels) after the sampling reduction.
    sampling : int
        Down sampling rate for component tree operation.

    Returns
    -------
    xpArray
        Area closed result.
    """
    return _generic_area_filtering(image=image, area_threshold=area_threshold, tree_type="min", sampling=sampling)


def area_white_top_hat(image: xpArray, area_threshold: float, sampling: int, clip_negatives: bool = True) -> xpArray:
    """Area white hot transform on the downsampled input.
    The white top hat transform is I - O(I), where O(.) is the area opening operator and I the original input.
    Note: Moves the data to the CPU.

    Parameters
    ----------
    image : xpArray
        Input grayscale image.
    area_threshold : float
        Threshold of area (number of pixels) after the sampling reduction.
    sampling : int
        Down sampling rate for component tree operation.
    clip_zeros : bool
        This operation can result to negative values that can be clipped given this parameter.

    Returns
    -------
    xpArray
        Area white top hat result.
    """
    xp = Backend.get_xp_module(image)

    opened = area_opening(image, area_threshold=area_threshold, sampling=sampling)

    signed_type = np.result_type(opened.dtype, image.dtype, np.byte)
    wth = image - opened.astype(signed_type)

    if not clip_negatives:
        return wth

    return xp.clip(wth, 0, None).astype(image.dtype)


def area_black_top_hat(image: xpArray, area_threshold: float, sampling: int) -> xpArray:
    """Area black hot transform on the downsampled input.
    The black top hat transform is C(I) - I, where C(.) is the area closing operator and I the original input.
    Note: Moves the data to the CPU.

    Parameters
    ----------
    image : xpArray
        Input grayscale image.
    area_threshold : float
        Threshold of area (number of pixels) after the sampling reduction.
    sampling : int
        Down sampling rate for component tree operation.

    Returns
    -------
    xpArray
        Area black top hat result.
    """
    closed = area_closing(image, area_threshold=area_threshold, sampling=sampling)
    return closed - image
