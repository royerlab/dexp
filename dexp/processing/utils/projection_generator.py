from typing import Callable, Iterator, Optional, Tuple

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def _get_projection_type(type: str, axis: Tuple[int]) -> Callable:
    xp = Backend.get_xp_module()
    projections = {
        "mean": lambda im: xp.mean(im, axis=axis),
        "median": lambda im: xp.median(im, axis=axis),
        "max": lambda im: xp.median(im, axis=axis),
        "min": lambda im: xp.min(im, axis=axis),
        "max-min": lambda im: xp.max(im, axis=axis) - xp.min(im, axis=axis),
    }
    try:
        return projections[type]
    except KeyError:
        raise ValueError(f"Unknown projection type: {type}")


def projection_generator(
    image: xpArray, axis_range: Optional[Tuple[int, ...]] = None, projection_type: str = "mean", nb_axis: int = 2
) -> Iterator[Tuple[int, int, xpArray]]:
    """
    Generates all nD projection of an image. Currently only supports 2D projections.

    Parameters
    ----------
    image: image to compute center of mass of.
    axis_range: Axis range to compute projections, can be: (n) for [0, n[, (m,n) for [m,n[, and (m,n,step)
        for {i*step | m<=i*step<n & i integer} projection_type: Projection type to use when in
        'projection' mode: 'mean', 'min', 'max', 'max-min'
    nb_axis: number of axis to project to, currently only supports 2D projections.

    Returns
    -------
    tuples of axis and projections: (u,v,..., projection)

    """
    if nb_axis != 2:
        # TODO: this could be fixed using itertools.combinations
        raise NotImplementedError("Not implemented for nb_axis!=2")

    ndim = image.ndim

    if axis_range is None:
        axis_range = (0, ndim, 1)

    for u in range(*axis_range):
        for v in range(*axis_range):
            if u < v:
                proj_axis = tuple(set(range(*axis_range)).difference({u, v}))
                projection = _get_projection_type(projection_type.lower(), proj_axis)
                projected_image = projection(image)
                projected_image = projected_image.astype(dtype=image.dtype, copy=False)

                yield u, v, projected_image

            elif u < 0 or v < 0 or u >= ndim or v >= ndim:
                raise ValueError(f"Axis range {axis_range} is out-of-bounds for the image of dimension: {ndim}")
