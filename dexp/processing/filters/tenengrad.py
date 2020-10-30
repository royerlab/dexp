from dexp.processing.backends.backend import Backend


def tenengrad(backend:Backend, image):
    xp = backend.get_xp_module(image)
    sp = backend.get_sp_module(image)
    ndim = image.ndim

    tenengrad_image=  xp.zeros_like(image)

    for i in range(ndim):
        tenengrad_image += sp.ndimage.sobel(image, axis=i) ** 2

    tenengrad_image = xp.sqrt(tenengrad_image)

    return tenengrad_image