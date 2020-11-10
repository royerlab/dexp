import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve


def lucy_richardson_deconvolution(backend: Backend,
                                   image,
                                   psf,
                                   max_num_iterations: int = 50,
                                   max_correction: float=8,
                                   power: int = 1,
                                   back_projection='tpsf',
                                   padding=0,
                                   padding_mode=None,
                                   convolve_method = fft_convolve,
                                   internal_dtype=numpy.float16):
    """
    Decponvolves an nD image given a point-spread-function.

    Parameters
    ----------
    backend : backend to use
    image : image to deconvolve
    psf : point-spread-function (must have the same number of dimensions as image!)
    max_number_iterations : max number of iterations
    power : power to elevate relative blur to
    back_projection : back projection operator to use.
    internal_dtype : dtype to use internally for computation.

    Returns
    -------
    Deconvolved image

    """

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if image.ndim != psf.ndim:
        raise ValueError("Two images must have same number of dimensions!")

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype, force_copy=True)
    psf = backend.to_backend(psf, dtype=internal_dtype, force_copy=True)

    back_projector = xp.flip(psf)

    #TODO: option?
    min_value = xp.min(image)
    max_value = xp.max(image)
    image -= min_value
    image *= 1 / (max_value - min_value)
    image = xp.clip(0, None, out=image)

    result = xp.full(
        image.shape, float(xp.mean(image))
    )

    psf_shape = psf.shape
    pad_width = tuple(
        (max(padding, (s - 1) // 2), max(padding, (s - 1) // 2))
        for s in psf_shape
    )

    for i in range(max_num_iterations):

        if padding > 0:
            padded_candidate_deconvolved_image = xp.pad(
                result,
                pad_width=pad_width,
                mode=padding_mode,
            )
        else:
            padded_candidate_deconvolved_image = result

        convolved = convolve_method(
            backend,
            padded_candidate_deconvolved_image,
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

        if padding:
            relative_blur = numpy.pad(
                relative_blur, pad_width=pad_width, mode=padding_mode
            )

        multiplicative_correction = convolve_method(
            backend,
            relative_blur,
            back_projector,
            mode='valid' if padding else 'same',
        )

        result *= multiplicative_correction

    result[result > 1] = 1
    result[result < 0] = 0

    result *= (max_value - min_value)
    result += min_value

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
