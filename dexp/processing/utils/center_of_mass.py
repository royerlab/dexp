from dexp.processing.backends.backend import Backend
from dexp.processing.utils.projection_generator import projection_generator


def center_of_mass(image,
                   mode: str = 'projection',
                   projection_type: str = 'mean',
                   remove_offset: bool = True,
                   offset_mode: str = 'min'):
    """
    Computes the center of mass of an image.

    Parameters
    ----------
    image: image to compute center of mass of.
    mode: Can be either 'full' or 'projection'. Full mode might lead to high memory consumption depending on the backend, projection mode gives some extra flexibility and is overall safer.
    projection_type: Projection type to use when in 'projection' mode: 'mean', 'min', 'max'
    remove_offset: removes offset to help remove the influence of the background on the center of mass
    offset_mode: Choice of offset to remove, can be either: 'min', 'median', 'mean'.

    Returns
    -------
    Center of mass as a vector of integers

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if mode == 'full':
        if remove_offset:
            image = _remove_offset(image, offset_mode, xp)

        com = sp.ndimage.center_of_mass(image)
        com = xp.asarray(com, dtype=xp.float32)
        return com
    elif mode == 'projection':
        ndim = image.ndim

        com = xp.zeros((ndim,), dtype=xp.float32)
        count = xp.zeros((ndim,), dtype=xp.float32)

        for u, v, projected_image in projection_generator(image):
            if remove_offset:
                projected_image = _remove_offset(projected_image, offset_mode, xp)

            du, dv = sp.ndimage.center_of_mass(projected_image)
            com[u] += du
            com[v] += dv
            count[u] += 1
            count[v] += 1

        com /= count

        return com


def _remove_offset(image, offset_mode, xp):
    if offset_mode == 'min':
        image -= image.min()
    elif offset_mode == 'median':
        image -= xp.median(image)
    elif offset_mode == 'mean':
        image -= xp.mean(image)
    elif offset_mode == 'middle':
        minv = image.min()
        maxv = image.max()
        image -= 0.5 * (maxv + minv)
    else:
        raise ValueError(f"Unknown offset mode: {offset_mode}")
    image = xp.clip(image, a_min=0, a_max=None, out=image)
    return image
