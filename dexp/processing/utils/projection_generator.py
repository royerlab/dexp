from typing import Tuple, Optional

from dexp.processing.backends.backend import Backend


def projection_generator(image,
                         axis_range: Optional[Tuple[int, ...]] = None,
                         projection_type: str = 'mean',
                         nb_axis: int = 2):
    """
    Generates all nD projection of an image. Currently only supports 2D projections.

    Parameters
    ----------
    image: image to compute center of mass of.
    axis_range: Axis range to compute projections, can be: (n) for [0, n[, (m,n) for [m,n[, and (m,n,step) for {i*step | m<=i*step<n & i integer}
    projection_type: Projection type to use when in 'projection' mode: 'mean', 'min', 'max', 'max-min'
    nb_axis: number of axis to project to, currently only supports 2D projections.

    Returns
    -------
    tuples of axis and projections: (u,v,..., projection)

    """
    xp = Backend.get_xp_module()

    if nb_axis != 2:
        raise NotImplementedError(f"Not implemented for nb_axis!=2")

    ndim = image.ndim

    if axis_range is None:
        axis_range = (0, ndim, 1)

    for u in range(*axis_range):
        for v in range(*axis_range):
            if u < v:
                proj_axis = tuple(set(range(*axis_range)).difference({u, v}))
                if projection_type == 'mean':
                    projected_image = xp.mean(image, axis=proj_axis)
                elif projection_type == 'median':
                    projected_image = xp.median(image, axis=proj_axis)
                elif projection_type == 'max':
                    projected_image = xp.max(image, axis=proj_axis)
                elif projection_type == 'min':
                    projected_image = xp.min(image, axis=proj_axis)
                elif projection_type == 'max-min':
                    projected_image = xp.max(image, axis=proj_axis)
                    projected_image -= xp.min(image, axis=proj_axis)
                else:
                    raise ValueError(f"Unknown projection type: {projection_type}")

                projected_image = projected_image.astype(dtype=image.dtype, copy=False)

                yield u, v, projected_image

            elif u < 0 or v < 0 or u >= ndim or v >= ndim:
                raise ValueError(f"Axis range {axis_range} is out-of-bounds for the image of dimension: {ndim}")
