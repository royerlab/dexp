import itertools

from dexp.processing.utils.projection_generator import projection_generator
from dexp.utils.backends import Backend


def center_of_mass(
    image,
    mode: str = "projection",
    projection_type: str = "mean",
    offset_mode: str = "min",
    bounding_box: bool = False,
):
    """
    Computes the center of mass of an image.

    Parameters
    ----------
    image: image to compute center of mass of.
    mode: Can be either 'full' or 'projection'. Full mode might lead to high memory consumption
        depending on the backend, projection mode gives some extra flexibility and is overall safer.
    projection_type: Projection type to use when in 'projection' type: 'mean', 'min', 'max', 'max-min'
    offset_mode: Choice of offset to remove, can be either: 'min', 'median', 'mean', 'middle',
        and for example: 'p=10' for removing and clipping the lower 10% percentile. Set to N one for no offset removal.
    remove_offset: removes offset to help remove the influence of the background on the center of mass
    bounding_box: if True, the center of mass of the bounding box of non-zero pixels is returned.

    Returns
    -------
    Center of mass as a vector of integers

    """
    xp = Backend.get_xp_module()

    if mode == "full":
        if offset_mode is not None:
            image = _remove_offset(image, offset_mode, xp)

        com = _center_of_mass(image, bounding_box)

    elif mode == "projection":
        ndim = image.ndim

        com = xp.zeros((ndim,), dtype=xp.float32)
        count = xp.zeros((ndim,), dtype=xp.float32)

        for u, v, projected_image in projection_generator(image, projection_type=projection_type, nb_axis=2):
            if offset_mode is not None:
                projected_image = _remove_offset(projected_image, offset_mode, xp)

            du, dv = _center_of_mass(projected_image, bounding_box)
            com[u] += du
            com[v] += dv
            count[u] += 1
            count[v] += 1

        com /= count

    com = xp.asarray(com, dtype=xp.float32)
    return com


def _center_of_mass(image, bounding_box):
    if bounding_box:
        bbox = _bounding_box(image)
        com = tuple(0.5 * bbox[2 * a] + 0.5 * bbox[2 * a + 1] for a in range(len(bbox) // 2))
    else:
        sp = Backend.get_sp_module()
        com = sp.ndimage.center_of_mass(image)

    return com


def _bounding_box(image):
    xp = Backend.get_xp_module(image)
    N = image.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = xp.any(image, axis=ax)
        out.extend(xp.where(nonzero)[0][[0, -1]])
    return tuple(out)


def _remove_offset(image, offset_mode, xp):
    if offset_mode == "min":
        image -= image.min()
    elif offset_mode.startswith("p="):
        percentile = float(offset_mode.split("=")[1].strip())
        image -= xp.percentile(image, percentile).astype(image.dtype, copy=False)
    elif offset_mode == "median":
        image -= xp.median(image).astype(image.dtype, copy=False)
    elif offset_mode == "mean":
        image -= xp.mean(image).astype(image.dtype, copy=False)
    elif offset_mode == "middle":
        minv = image.min()
        maxv = image.max()
        offset = 0.5 * (maxv + minv)
        offset = offset.astype(image.dtype)
        image -= offset
    else:
        raise ValueError(f"Unknown offset mode: {offset_mode}")
    image = xp.clip(image, a_min=0, a_max=None, out=image)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')

    return image
