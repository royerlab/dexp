import math
from typing import Optional, Tuple, Union

import numpy

from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd
from dexp.processing.filters.kernels.wiener_butterworth import wiener_butterworth_kernel
from dexp.processing.utils.nan_to_zero import nan_to_zero
from dexp.processing.utils.normalise import Normalise
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def lucy_richardson_deconvolution(
    image: xpArray,
    psf: xpArray,
    num_iterations: Optional[int] = None,
    max_correction: Optional[float] = None,
    power: float = 1.0,
    back_projection: Optional[str] = None,
    wb_cutoffs: Union[float, Tuple[float, ...], None] = 0.9,
    wb_beta: float = 0.05,
    wb_order: int = 2,
    padding: int = 0,
    padding_mode: str = "edge",
    normalise_input: bool = True,
    normalise_minmax: Optional[Tuple[float, float]] = None,
    clip_output: bool = False,
    blind_spot: int = 0,
    blind_spot_mode: str = "median+uniform",
    blind_spot_axis_exclusion: Optional[Union[str, Tuple[int, ...]]] = None,
    eps: float = 1e-12,
    convolve_method=fft_convolve,
    internal_dtype=None,
):
    """
    Deconvolves an nD image given a point-spread-function.

    Parameters
    ----------
    image : image to deconvolve
    psf : point-spread-function (must have the same number of dimensions as image!)
    num_iterations : number of iterations
    max_correction : Lucy-Richardson correction will remain clamped within [1/mc, mc] (before back projection)
    power : power to elevate coorection (before back projection)
    back_projection : back projection operator to use: 'tpsf' or 'wb'.
    wb_cutoffs : Wiener-Butterworth cutoffs for wb back projection.
    wb_beta : Wiener-Butterworth backprojection beta parameter.
    wb_order : Wiener-Butterworth backprojection order parameter.
    padding : padding (see numpy/cupy pad function)
    padding_mode : padding mode (see numpy/cupy pad function)
    normalise_input : This deconvolution code assumes values within [0, 1], by default input images
        are normalised to that range, but if already normalised, then normalisation can be ommited.
    normalise_minmax : Use the given tuple (min, max) for normalisation
    clip_output : Clip output to input range, or not
    blind_spot : If zero, blind-spot is disabled. If blind_spot>0 it is active and the integer represents
        the blind-spot kernel support. A value of 3 or 5 are good and help reduce the impact of noise on deconvolution.
    blind_spot_mode : blind-spot mode, can be 'mean' or 'median'
    blind_spot_axis_exclusion : if None no axis is excluded from the blind-spot support kernel, otherwise
        if a tuple of ints is provided, these refer to the then the support kernel is clipped along these axis.
        For example for a 3D stack where the sampling along z (first axis) is poor,
        use: (0,) so that blind-spot kernel does not extend in z.
    eps: epsilon to avoid dividing by zero
    convolve_method : convolution method to use
    internal_dtype : dtype to use internally for computation.

    Returns
    -------
    Deconvolved image

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if image.ndim != psf.ndim:
        raise ValueError("The image and PSF must have same number of dimensions!")

    if internal_dtype is None:
        internal_dtype = numpy.float32

    if isinstance(Backend.current(), NumpyBackend):
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype)
    psf = Backend.to_backend(psf, dtype=internal_dtype)

    if blind_spot > 0:
        if 2 * (blind_spot // 2) == blind_spot:
            raise ValueError(f"Blind spot size must be an odd integer, blind_spot={blind_spot} is not!")

        if "gaussian" in blind_spot_mode:
            full_kernel = gaussian_kernel_nd(ndim=image.ndim, size=blind_spot, sigma=max(1, blind_spot // 2))
        else:
            full_kernel = xp.ones(shape=(blind_spot,) * image.ndim, dtype=internal_dtype)

        if blind_spot_axis_exclusion is not None:
            c = blind_spot // 2

            slicing_n = [
                slice(None, None, None),
            ] * image.ndim
            slicing_p = [
                slice(None, None, None),
            ] * image.ndim
            for axis in blind_spot_axis_exclusion:
                slicing_n[axis] = slice(0, c, None)
                slicing_p[axis] = slice(c + 1, blind_spot, None)

            full_kernel[slicing_n] = 0
            full_kernel[slicing_p] = 0

        full_kernel /= full_kernel.sum()
        donut_kernel = full_kernel.copy()
        donut_kernel[(slice(blind_spot // 2, blind_spot // 2 + 1, None),) * image.ndim] = 0
        donut_kernel /= donut_kernel.sum()
        # psf_original = psf.copy()
        psf = sp.ndimage.convolve(psf, donut_kernel)
        psf /= psf.sum()

        if "median" in blind_spot_mode:
            image = sp.ndimage.median_filter(image, footprint=full_kernel)
        elif "mean" in blind_spot_mode:
            image = sp.ndimage.convolve(image, donut_kernel)

        # from napari import Viewer, gui_qt
        # with gui_qt():
        #     def _c(array):
        #         return backend.to_numpy(array)
        #
        #     viewer = Viewer()
        #     #viewer.add_image(_c(image), name='image')
        #     viewer.add_image(_c(psf_original), name='psf_original')
        #     viewer.add_image(_c(donut_kernel), name='donut_kernel', color=False)
        #     viewer.add_image(_c(psf), name='psf')
        #     viewer.add_image(_c(sp.ndimage.convolve(psf, full_kernel)), name='psf_for_backproj')
        #     viewer.add_image(_c(back_projector), name='psf')

    # Default back projection:
    back_projection = "tpsf" if back_projection is None else back_projection

    # Back projection setting:
    if back_projection == "tpsf":
        back_projector = xp.flip(psf)
    elif back_projection == "wb":
        back_projector = wiener_butterworth_kernel(
            kernel=xp.flip(psf), cutoffs=wb_cutoffs, beta=wb_beta, order=wb_order, dtype=xp.float64
        )
        back_projector = back_projector.astype(psf.dtype)
    else:
        raise ValueError(f"back projection mode: {back_projection} not supported.")

    # Default number of iterations:
    if num_iterations is None:
        if back_projection == "tpsf":
            num_iterations = 20
        elif back_projection == "wb":
            num_iterations = 3

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     psf_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(psf))))
    #     back_projector_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(back_projector))))
    #     viewer = Viewer()
    #     viewer.add_image(_c(psf), name='psf')
    #     viewer.add_image(_c(back_projector), name='back_projector')
    #     viewer.add_image(_c(psf_f), name='psf_f', colormap='viridis')
    #     viewer.add_image(_c(back_projector_f), name='back_projector_f', colormap='viridis')

    # Normalisation:
    normalise = Normalise(
        image, minmax=normalise_minmax, do_normalise=normalise_input, clip=False, dtype=internal_dtype
    )

    image = normalise.forward(image)

    # Padding:
    if padding > 0:
        image = numpy.pad(image, pad_width=padding, mode=padding_mode)

    # Result array:
    result = xp.full(image.shape, float(xp.mean(image)), dtype=internal_dtype)

    # LR iterations:
    for i in range(num_iterations):
        # print(f"LR iteration: {i}")
        # Convolution with PSF:
        convolved = convolve_method(
            result,
            psf,
        )
        # convolved = xp.clip(convolved, a_min=0, a_max=None, out=convolved)
        # Computes relative blur:
        relative_blur = (image + eps) / (convolved + eps)
        # replace Nans with zeros, and +inf with very large values:
        relative_blur = nan_to_zero(relative_blur, copy=False)
        # Limits max correction:
        if max_correction is not None:
            relative_blur[relative_blur > max_correction] = max_correction
            relative_blur[relative_blur < 1 / max_correction] = 1 / max_correction

        # Back-projection:
        multiplicative_correction = convolve_method(
            relative_blur,
            back_projector,
        )

        # Multiplicative correction can be optionally elevated to a power:
        if power != 1.0:
            multiplicative_correction = xp.clip(multiplicative_correction, 0, None, out=multiplicative_correction)
            multiplicative_correction **= 1 + (power - 1) / (math.sqrt(1 + i))

        # Apply multiplicative correction:
        result *= multiplicative_correction

    # Delete intermediates:
    del multiplicative_correction, relative_blur, back_projector, convolved, psf

    # Clips output:
    if clip_output:
        if normalise_input:
            result = xp.clip(result, 0, 1, out=result)
        else:
            result = xp.clip(result, xp.min(image), xp.max(image), out=result)

    # Denormalises result:
    result = normalise.backward(result)

    # Removes padding:
    if padding > 0:
        slicing = (slice(padding, -padding),) * result.ndim
        result = result[slicing]

    # converts to original dtype:
    result = result.astype(original_dtype, copy=False)

    # from napari import Viewer
    # import napari
    # with napari.gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image1), name='image_1')
    #     viewer.add_image(_c(image1), name='image_2')

    return result
