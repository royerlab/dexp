import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.utils.element_wise_affine import element_wise_affine


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

    if back_projection == 'tpsf':
        back_projector = xp.flip(psf)
    else:
        raise ValueError(f"back projection mode: {back_projection} not supported.")

    if normalise_input:
        min_value = xp.min(image)
        max_value = xp.max(image)
        alpha = (1 / (max_value - min_value)).astype(internal_dtype)
        image = element_wise_affine(backend, image, alpha, -min_value)
        image = xp.clip(image, 0, 1, out=image)

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

        # zeros = convolved == 0
        # relative_blur[zeros] = 0

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

    if normalise_input:
        result = element_wise_affine(backend, result, (max_value - min_value), min_value)

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
