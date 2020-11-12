import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.filters.kernels import gaussian_kernel_nd
from dexp.processing.utils.element_wise_affine import element_wise_affine
from dexp.processing.utils.nan_to_num import nan_to_zero
from dexp.processing.utils.normalise import normalise


def lucy_richardson_deconvolution(backend: Backend,
                                  image,
                                  psf,
                                  num_iterations: int = 50,
                                  max_correction: float = 8,
                                  power: float = 1.0,
                                  back_projection='tpsf',
                                  padding: int = 0,
                                  padding_mode: str = 'edge',
                                  normalise_input: bool = True,
                                  clip_output: bool = True,
                                  blind_spot: int = 0,
                                  convolve_method=fft_convolve,
                                  internal_dtype=numpy.float16):
    """
    Decponvolves an nD image given a point-spread-function.

    Parameters
    ----------
    backend : backend to use
    image : image to deconvolve
    psf : point-spread-function (must have the same number of dimensions as image!)
    num_iterations : number of iterations
    max_correction : Lucy-Richardson correction will remain clamped within [1/mc, mc] (before back projection)
    power : power to elevate coorection (before back projection)
    back_projection : back projection operator to use.
    padding : padding (see numpy/cupy pad function)
    padding_mode : padding mode (see numpy/cupy pad function)
    normalise_input : This deconvolution code assumes values within [0, 1], by default input images are normalised to that range, but if already normalised, then normalisation can be ommited.
    blind_spot : If zero, blind-spot is disabled. If blind_spot>0 it si active and the integer represents the blind-spot kernel support. A value of 3 or 5 are good and help reduce the impact of noise on deconvolution.
    convolve_method : convolution method to use
    clip_output : By default the output is clipped to the input image range.
    internal_dtype : dtype to use internally for computation.

    Returns
    -------
    Deconvolved image

    """

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if image.ndim != psf.ndim:
        raise ValueError("The image and PSF must have same number of dimensions!")

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype)
    psf = backend.to_backend(psf, dtype=internal_dtype)

    if blind_spot > 0:
        if 2*(blind_spot//2) == blind_spot:
            raise ValueError(f"Blind spot size must be an odd integer, blind_spot={blind_spot} is not!")
        full_kernel = xp.ones(shape=(blind_spot,)*image.ndim, dtype=internal_dtype)
        # some experiments suggests that a uniform donut filter works better
        full_kernel = gaussian_kernel_nd(backend, ndim=image.ndim, size=blind_spot, sigma=1)
        full_kernel = full_kernel / full_kernel.sum()
        donut_kernel = full_kernel.copy()
        donut_kernel[(slice(blind_spot//2, blind_spot//2+1, None),)*image.ndim] = 0
        donut_kernel = donut_kernel / donut_kernel.sum()
        #psf_original = psf.copy()
        psf = sp.ndimage.convolve(psf, donut_kernel)
        psf = psf / psf.sum()
        image = fft_convolve(backend, image, donut_kernel)


        # from napari import Viewer, gui_qt
        # with gui_qt():
        #     def _c(array):
        #         return backend.to_numpy(array)
        #
        #     viewer = Viewer()
        #     #viewer.add_image(_c(image), name='image')
        #     viewer.add_image(_c(psf_original), name='psf_original')
        #     viewer.add_image(_c(donut_kernel), name='donut_kernel', rgb=False)
        #     viewer.add_image(_c(psf), name='psf')
        #     viewer.add_image(_c(sp.ndimage.convolve(psf, full_kernel)), name='psf_for_backproj')
        #     viewer.add_image(_c(back_projector), name='psf')


    if back_projection == 'tpsf':
        back_projector = xp.flip(psf.copy())
    else:
        raise ValueError(f"back projection mode: {back_projection} not supported.")

    # if blind_spot:
    #     center = tuple(s//2 for s in back_projector.shape)
    #     slicing = tuple(slice(c, c+1, None) for c in center)
    #     psf[slicing] = 0
    #     psf = psf / psf.sum()
    #     # back_projector[slicing] = 0
    #     # back_projector = back_projector / back_projector.sum()

    if normalise_input:
        image, denorm_fun = normalise(backend, image, out=image)
    else:
        denorm_fun = None

    if padding > 0:
        image = numpy.pad(image, pad_width=padding, mode=padding_mode)

    result = xp.full(
        image.shape,
        float(xp.mean(image)),
        dtype=internal_dtype
    )

    for i in range(num_iterations):

        convolved = convolve_method(
            backend,
            result,
            psf,
            mode='valid' if padding else 'same',
        )

        relative_blur = image / convolved

        # replace Nans with zeros:
        #zeros = convolved == 0 #relative_blur[zeros] = 0
        relative_blur = nan_to_zero(backend, relative_blur, copy=False)

        relative_blur[
            relative_blur > max_correction
            ] = max_correction
        relative_blur[relative_blur < 1 / max_correction] = (
                1 / max_correction
        )

        if power != 1.0:
            relative_blur **= power

        multiplicative_correction = convolve_method(
            backend,
            relative_blur,
            back_projector,
            mode='valid' if padding else 'same',
        )

        result *= multiplicative_correction

    del multiplicative_correction, relative_blur, back_projector, convolved, psf

    if clip_output:
        if normalise_input:
            result = xp.clip(result, 0, 1, out=result)
        else:
            result = xp.clip(result, xp.min(image), xp.max(image), out=result)

    if denorm_fun:
        result = denorm_fun(result)

    if padding > 0:
        slicing = (slice(padding, -padding),) * result.ndim
        result = result[slicing]

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
