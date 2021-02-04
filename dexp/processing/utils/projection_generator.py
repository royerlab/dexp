from dexp.processing.backends.backend import Backend


def projection_generator(image, projection_type: str = 'mean', nb_axis: int = 2):
    """
    Generates all nD projection of an image. Currently only supports 2D projections.

    Parameters
    ----------
    image: image to compute center of mass of.
    projection_type: Projection type to use when in 'projection' mode: 'mean', 'min', 'max'
    nb_axis: number of axis to project to, currently only supports 2D projections.

    Returns
    -------
    tuples of axis and projections: (u,v,..., projection)

    """
    xp = Backend.get_xp_module()

    ndim = image.ndim

    for u in range(ndim):
        for v in range(ndim):
            if u < v:
                proj_axis = tuple(set(range(ndim)).difference({u, v}))
                if projection_type == 'mean':
                    projected_image = xp.mean(image, axis=proj_axis)
                elif projection_type == 'median':
                    projected_image = xp.median(image, axis=proj_axis)
                elif projection_type == 'max':
                    projected_image = xp.max(image, axis=proj_axis)
                elif projection_type == 'min':
                    projected_image = xp.min(image, axis=proj_axis)
                else:
                    raise ValueError(f"Unknown projection type: {projection_type}")

                yield u, v, projected_image

