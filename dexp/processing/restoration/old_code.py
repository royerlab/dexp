def fast_hybrid_filter(backend: Backend, image, size, sigma=1):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image = image.astype(numpy.float32, copy=False)

    filter_size = 3
    nb_steps = (size // filter_size) // 2

    # print(f'nb_steps={nb_steps}, filter_size={filter_size}, sigma={sigma}')
    temp1 = backend.to_backend(image)
    temp2 = xp.empty_like(numpy.zeros(image.shape, image.dtype))
    for step in range(nb_steps):
        # print(f'step={step+1}')
        sp.ndimage.filters.median_filter(temp1, size=filter_size, out=temp2)
        sp.ndimage.filters.gaussian_filter(temp2, sigma=sigma, out=temp1)

    return temp1.get()


def median_dehaze(backend: Backend, image, size=20):
    sp = backend.get_sp_module()
    xp = backend.get_xp_module()
    image = backend.to_backend(image)
    median_filtered = sp.ndimage.filters.median_filter(image, size)
    median_sharpened = xp.clip(image.astype(numpy.float16) - median_filtered, 0, numpy.math.inf)
    return median_sharpened, median_filtered


def gaussian_dehaze(backend: Backend, image, size=20):
    sp = backend.get_sp_module()
    xp = backend.get_xp_module()
    image = backend.to_backend(image)
    # Three sigmas is the support of the filter
    gaussian_filtered = sp.ndimage.filters.gaussian_filter(image, sigma=size // 3)
    gaussian_sharpened = xp.clip(image - gaussian_filtered, 0, numpy.math.inf)
    return gaussian_sharpened, gaussian_filtered


def hybrid_dehazing(backend: Backend, image, size=20, sigma=2):
    xp = backend.get_xp_module()
    image = backend.to_backend(image)
    hybrid_filtered = fast_hybrid_filter(backend, image, size, sigma)
    hybrid_sharpened = xp.clip(image.astype(numpy.float16) - hybrid_filtered, 0, numpy.math.inf)
    return hybrid_sharpened, hybrid_filtered


def dehaze(backend: Backend,
           image,
           mode='hybrid',
           size=20,
           normalise_range=True, min=None, max=None, epsilon=0.1,
           clear_margins=True, margin_pad=True, margin=8,
           denoise=False
           ):
    sp = backend.get_sp_module()
    xp = backend.get_xp_module()
    image = backend.to_backend(image)

    if mode == 'gaussian':
        sharpened, _ = gaussian_dehaze(backend, image, size)

    if mode == 'median':
        sharpened, _ = median_dehaze(backend, image, size)

    if mode == 'hybrid':
        sharpened, _ = hybrid_dehazing(backend, image, size)

    if normalise_range:
        if not min is None:
            org_min = min
        else:
            org_min = xp.percentile(image, epsilon)

        if not max is None:
            org_max = max
        else:
            org_max = xp.percentile(image, 100 - epsilon)

        new_min = xp.percentile(sharpened, epsilon)
        new_max = xp.percentile(sharpened, 100 - epsilon)

        scaling = (org_max - org_min) / (new_max - new_min)
        offset = org_min - scaling * new_min

        sharpened = sharpened * scaling + offset

    if clear_margins:
        sharpened = sharpened[margin:-margin, margin:-margin, margin:-margin]
        if margin_pad:
            sharpened = xp.pad(sharpened, pad_width=margin, mode='constant')

    if denoise:
        sharpened = sharpened.astype(numpy.uint16, copy=False)
        sharpened = sp.ndimage.filters.median_filter(sharpened, size=3)

    return sharpened
