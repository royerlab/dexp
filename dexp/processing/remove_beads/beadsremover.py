from typing import List, Optional, Tuple, Union

import numpy as np
from arbol import aprint
from numpy.typing import ArrayLike
from scipy.signal._signaltools import _centered

from dexp.utils import xpArray
from dexp.utils.backends import Backend


class Bead:
    def __init__(self, image: ArrayLike, coord: Optional[Tuple[int]], size: int):
        if coord is not None:
            self.slicing = self.get_slice(coord, size)
            self.data = image[self.slicing]
        else:
            self.slicing = None
            self.data = image
        self.data = Backend.current().to_numpy(self.data)
        self.data = self.data / np.linalg.norm(self.data)

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    @staticmethod
    def get_slice(coord: Tuple[int], size: int) -> List[slice]:
        slicing = tuple(slice(c - size // 2, c + size // 2 + 1) for c in coord)
        return slicing

    @staticmethod
    def from_data(array: ArrayLike) -> "Bead":
        return Bead(array, None, -1)

    def __array__(self, *args) -> ArrayLike:
        return self.data.reshape(-1)


class BeadsRemover:
    def __init__(self, peak_threshold: int, similarity_threshold: float, psf_size: int = 31, verbose: bool = False):
        """
        Helper object to detect beads, extract the PSF from it and remove it if necessary.

        Args:
            peak_threshold (int): Peak maxima threshold for object (beads) detection.
                Lower values detect more objects, but it might get more false positives.
            similarity_threshold (float): Threshold for beads selection given their cosine
                similarity with the estimated (median) PSF.
            psf_size (int, optional): PSF size (shape). Defaults to 31.
            verbose (bool, optional): Flag to display intermediate results. Defaults to False.
        """
        self.peak_threshold = peak_threshold
        self.similarity_threshold = similarity_threshold
        self.psf_size = psf_size
        self.beads: List[Bead] = []

        self.verbose = verbose

    def detect_beads(self, array: xpArray, estimated_psf: Optional[ArrayLike] = None) -> ArrayLike:
        from cucim.skimage.feature import peak_local_max

        backend = Backend.current()

        sp = backend.get_sp_module()
        xp = backend.get_xp_module()

        if estimated_psf is None:
            aprint("Finding beads with point detection.")
            kernel = xp.zeros((1, 5, 5))
            kernel[0, 2, 2] = 4
            kernel[0, 2, 0] = -1
            kernel[0, 0, 2] = -1
            kernel[0, 2, 4] = -1
            kernel[0, 4, 2] = -1
        else:
            aprint("Finding beads with estimated PSF.")
            kernel = estimated_psf.copy()
            dark_region = kernel < kernel.mean()
            bright_region = np.logical_not(dark_region)
            # equalize the values of both the regions to be 1.0
            kernel[dark_region] = -1.0 / dark_region.sum()
            kernel[bright_region] /= kernel[bright_region].sum()

            kernel = backend.to_backend(kernel.astype(xp.float32))

        # try detecting beads with given PSF or dot detection
        filtered = sp.ndimage.correlate(array.astype(xp.float32), weights=kernel)

        # small blur to center detection
        mean_kernel = xp.ones((5,) * array.ndim, dtype=xp.float32)
        mean_kernel /= mean_kernel.sum()
        filtered = sp.ndimage.correlate(filtered, weights=mean_kernel)

        # find local maxima in filtered image
        coordinates = peak_local_max(filtered, min_distance=self.psf_size, threshold_abs=500).get()

        # find local maxima in original image near the filtered maxima
        peaks = xp.zeros(array.shape, dtype=bool)
        peaks[tuple(coordinates.T)] = True
        peaks = sp.ndimage.binary_dilation(peaks, xp.ones((self.psf_size,) * array.ndim, dtype=bool))
        coordinates = peak_local_max(array, min_distance=self.psf_size, labels=peaks).get()

        if self.verbose:
            import napari

            viewer = napari.Viewer()
            viewer.add_image(
                filtered.get(), name="filtered", blending="additive", colormap="red", contrast_limits=(0, 5000)
            )
            viewer.add_image(
                array.get(), name="original", blending="additive", colormap="green", contrast_limits=(0, 5000)
            )
            peaks = xp.zeros(array.shape, dtype=bool)
            peaks[tuple(coordinates.T)] = True
            viewer.add_image(peaks.get(), name="peaks", blending="additive", colormap="gray", contrast_limits=(0, 1))
            napari.run()

        array = array.get()
        beads = []
        for coord in coordinates:
            bead = Bead(array, coord, self.psf_size)
            if np.all(np.equal(bead.shape, self.psf_size)):
                beads.append(bead)

        if len(beads) == 0:
            avg_bead = np.zeros((self.psf_size,) * array.ndim)
        else:
            avg_bead = Bead.from_data(np.median(np.stack(tuple(beads)), axis=0).reshape((self.psf_size,) * array.ndim))

        # select regions that are similar to the median estimated bead
        selected_beads = []
        for bead in beads:
            angle = np.dot(bead, avg_bead)
            if angle > self.similarity_threshold:
                selected_beads.append(bead)
                array[bead.slicing] = 255

        aprint(f"Detected {len(selected_beads)} beads.")

        if self.verbose:
            import napari

            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(avg_bead.data, name="estimated PSF", interpolation="nearest")
            viewer.add_image(array, contrast_limits=(0, 500), name="marked beads")
            napari.run()

        self.beads = selected_beads
        return avg_bead.data

    def remove_beads(self, array: xpArray, fill_value: int = 0) -> xpArray:
        for bead in self.beads:
            array[bead.slicing] = fill_value
        return array


def remove_beads_by_threshold(
    image: xpArray, psf_size: Union[int, Tuple[int]] = 35, window: int = 7, k: float = 0.01
) -> xpArray:
    """
    This functions estimates beads by a local threshold and
    replace them with the minimum value of their respective region

    Parameters
    ----------
    image : xpArray
        Input image, validated with raw data
    psf_size : Union[int, Tuple[int]], optional
        Estimated PSF size for removal after detection, by default 35
    window : int, optional
        Local threshold window, by default 15
    k : float, optional
        Local threshold std deviation parameter, by default 0.05
        Reference: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_sauvola

    Returns
    -------
    xpArray
        Input image without beads
    """
    from cucim.skimage.filters import threshold_niblack

    sp = Backend.get_sp_module(image)
    xp = Backend.get_xp_module(image)

    shape = image.shape
    sampling = 4

    # downsample and remove noise from image
    small_image = sp.ndimage.zoom(image, zoom=1 / sampling, order=1)
    small_image = sp.ndimage.convolve(small_image, xp.ones((3,) * image.ndim))

    # apply threshold and upsample
    threshold = threshold_niblack(small_image, window, k)
    threshold = sp.ndimage.zoom(threshold, zoom=sampling, order=1)
    threshold = _centered(threshold, shape)

    # detect beads
    detection = image > threshold
    detection = sp.ndimage.minimum_filter(detection, 2)  # erosion

    # dilate and fill regions
    detection = sp.ndimage.maximum_filter(detection, psf_size)  # dilation

    # replacing mask with minimum
    min_image = sp.ndimage.minimum_filter(image, psf_size)
    output = image.copy()
    output[detection] = min_image[detection]

    return output
