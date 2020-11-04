import numpy

from dexp.processing.backends.backend import Backend


def butterworth_kernel(backend: Backend, filter_shape, cutoffs, n=3):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    ndim = len(filter_shape)

    if ndim == 2:

        ly, lx = filter_shape
        cy, cx = cutoffs

        x = xp.linspace(-0.5, 0.5, lx)
        y = xp.linspace(-0.5, 0.5, ly)

        # An array with every pixel = radius relative to center
        radius = xp.sqrt(((x / cx) ** 2)[xp.newaxis, xp.newaxis, :] + ((y / cy) ** 2)[xp.newaxis, :, xp.newaxis])

    elif ndim == 3:
        lz, ly, lx = filter_shape
        cz, cy, cx = cutoffs

        x = xp.linspace(-0.5, 0.5, lx)
        y = xp.linspace(-0.5, 0.5, ly)
        z = xp.linspace(-0.5, 0.5, lz)

        # An array with every pixel = radius relative to center
        radius = xp.sqrt(((x / cx) ** 2)[xp.newaxis, xp.newaxis, :] + ((y / cy) ** 2)[xp.newaxis, :, xp.newaxis] + ((z / cz) ** 2)[:, xp.newaxis, xp.newaxis])

    filter = 1 / (1.0 + radius ** (2 * n))

    kernel = sp.fft.fftshift(xp.real(sp.fft.ifftn(sp.fft.ifftshift(filter))))

    kernel = kernel / kernel.sum()
    kernel = xp.squeeze(kernel)

    return kernel.astype(numpy.float32)


def butterworth_filter(backend: Backend, image, filter_shape=None, cutoffs=None, n=3, mode='reflect'):
    sp = backend.get_sp_module()

    if filter_shape is None:
        filter_shape = (11,) * image.ndim
    elif type(filter_shape) is int and image.ndim > 1:
        filter_shape = (filter_shape,) * image.ndim

    if cutoffs is None:
        cutoffs = (0.5,) * image.ndim
    elif type(cutoffs) is float and image.ndim > 1:
        cutoffs = (cutoffs,) * image.ndim

    butterworth_filter_numpy = butterworth_kernel(backend, filter_shape, cutoffs, n)
    butterworth_filter = backend.to_backend(butterworth_filter_numpy)

    image = backend.to_backend(image)
    filtered_image = sp.ndimage.convolve(image, butterworth_filter, mode=mode)

    return filtered_image
