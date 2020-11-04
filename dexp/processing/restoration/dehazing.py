from dexp.processing.backends.backend import Backend


def dehaze(backend: Backend,
           image,
           size=20
           ):
    sp = backend.get_sp_module()
    xp = backend.get_xp_module()
    image = backend.to_backend(image)

    # get rid of low values due to noise:
    image_zero_level = sp.ndimage.filters.maximum_filter(image, size=3)

    # find min values:
    image_zero_level = sp.ndimage.filters.minimum_filter(image_zero_level, size=size)

    # expand reach of these min values:
    image_zero_level = sp.ndimage.filters.maximum_filter(image_zero_level, size=size)

    # smooth out:
    image_zero_level = sp.ndimage.filters.gaussian_filter(image_zero_level, sigma=2)

    # remove zero level:
    dehazed_image = xp.maximum(image - image_zero_level, 0)

    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(image_zero_level, name='image_zero_level')
    #     viewer.add_image(dehazed_image, name='dehazed')

    return dehazed_image
