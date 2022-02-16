from functools import partial
from typing import Optional

import numpy as np
from scipy.signal._signaltools import _centered

from dexp.processing.morphology.utils import get_3d_image_graph
from dexp.processing.utils import apply_func
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
    import higra as hg

    if tree_type.lower() == "min":
        component_tree_fun = hg.component_tree_min_tree
    elif tree_type.lower() == "max":
        component_tree_fun = hg.component_tree_max_tree
    else:
        raise NotImplementedError

    orig_shape = image.shape
    sp = Backend.get_sp_module(image)
    if sampling != 1:
        image = sp.ndimage.zoom(image, zoom=1 / sampling, order=1)

    orig_dtype = image.dtype
    image = Backend.to_numpy(image, dtype=np.float32)

    if image.ndim == 2:
        graph = hg.get_4_adjacency_graph(image.shape)
    elif image.ndim == 3:
        graph = get_3d_image_graph(image.shape)
    else:
        raise NotImplementedError

    tree, alt = component_tree_fun(graph, image)

    area = hg.attribute_area(tree)
    filtered = hg.reconstruct_leaf_data(tree, alt, area < area_threshold)

    hg.clear_auto_cache()

    filtered = Backend.to_backend(filtered, dtype=orig_dtype)
    if sampling != 1:
        filtered = sp.ndimage.zoom(filtered, zoom=sampling, order=1)

    return _centered(filtered, orig_shape)


def area_opening(
    image: xpArray,
    area_threshold: float,
    sampling: int,
    axis: Optional[int] = None,
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
    axis : int
        Apply transformation of this axis individually.

    Returns
    -------
    xpArray
        Area opened result.
    """
    _area_opening = partial(_generic_area_filtering, area_threshold=area_threshold, tree_type="max", sampling=sampling)

    if axis is None:
        return _area_opening(image=image)

    return apply_func(image, func=_area_opening, axis=axis)


def area_closing(
    image: xpArray,
    area_threshold: float,
    sampling: int,
    axis: Optional[int] = None,
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
    axis : int
        Apply transformation of this axis individually.

    Returns
    -------
    xpArray
        Area closed result.
    """
    _area_closing = partial(_generic_area_filtering, area_threshold=area_threshold, tree_type="min", sampling=sampling)

    if axis is None:
        return _area_closing(image=image)

    return apply_func(image, func=_area_closing, axis=axis)


def area_white_top_hat(
    image: xpArray, area_threshold: float, sampling: int, clip_negatives: bool = True, axis: Optional[int] = None
) -> xpArray:

    """Area white hot transform on the downsampled input.
    This operation returns every *bright* component smaller than the given `area_threshold`.
    The white top hat transform is I - O(I), where O(.) is the area opening operator and I the original input.
    Note: It moves the data to the CPU.

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
    axis : int
        Apply transformation of this axis individually.

    Returns
    -------
    xpArray
        Area white top hat result.
    """
    xp = Backend.get_xp_module(image)

    opened = area_opening(image, area_threshold=area_threshold, sampling=sampling, axis=axis)

    signed_type = np.result_type(opened.dtype, image.dtype, np.byte)
    wth = image - opened.astype(signed_type)

    if not clip_negatives:
        return wth

    return xp.clip(wth, 0, None).astype(image.dtype)


def area_black_top_hat(image: xpArray, area_threshold: float, sampling: int, axis: Optional[int] = None) -> xpArray:
    """Area black hot transform on the downsampled input.
    This operation returns the complement (inverse mapping) of every *dark* component smaller than
    the given `area_threshold`.
    The black top hat transform is C(I) - I, where C(.) is the area closing operator and I the original input.
    Note: It moves the data to the CPU.

    Parameters
    ----------
    image : xpArray
        Input grayscale image.
    area_threshold : float
        Threshold of area (number of pixels) after the sampling reduction.
    sampling : int
        Down sampling rate for component tree operation.
    axis : int
        Apply transformation of this axis individually.

    Returns
    -------
    xpArray
        Area black top hat result.
    """
    closed = area_closing(image, area_threshold=area_threshold, sampling=sampling, axis=axis)
    return closed - image
