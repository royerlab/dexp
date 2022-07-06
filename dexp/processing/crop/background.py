from dexp.processing.utils.pad import fit_to_shape
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def foreground_mask(
    array: xpArray,
    intensity_threshold: int = 10,
    binary_area_threshold: int = 3000,
    downsample: int = 2,
    display: bool = False,
) -> xpArray:
    """
    Detects foreground by merging nearby components and removing objects below a given area threshold.

    Parameters
    ----------
    array : xpArray
        Input grayscale image.
    intensity_threshold : int, optional
        Threshold used to binarized filtered image --- detect objects.
    binary_area_threshold : int, optional
        Threshold used to remove detected objects.
    downsample : int, optional
        Downsample factor to speed up computation, it affects the distance between components and the area threshold.
    display : bool, optional
        Debugging flag, if true displays result using napari, by default False

    Returns
    -------
    xpArray
        Foreground mask.
    """

    ndi = Backend.get_sp_module(array).ndimage
    morph = Backend.get_skimage_submodule("morphology", array)

    ndim = array.ndim
    small = ndi.zoom(array, (1 / downsample,) * array.ndim, order=1)

    small = ndi.grey_opening(small, ndim * (3,))
    small = ndi.grey_closing(small, ndim * (7,))
    small = small > intensity_threshold
    small = ndi.grey_dilation(small, ndim * (9,))

    small = morph.remove_small_objects(small, binary_area_threshold)
    mask = ndi.zoom(small, (downsample,) * ndim, order=0)
    mask = fit_to_shape(mask, array.shape)

    if display:
        import napari

        v = napari.Viewer()
        v.add_image(Backend.to_numpy(array))
        v.add_labels(Backend.to_numpy(mask))

        napari.run()

    return mask
