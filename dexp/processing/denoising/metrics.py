from dexp.utils.backends import Backend


def mean_squared_error(image_a, image_b):
    # Backend:
    xp = Backend.get_xp_module(image_a)
    return xp.mean(xp.square(image_a - image_b))


def mean_absolute_error(image_a, image_b):
    # Backend:
    xp = Backend.get_xp_module(image_a)
    return xp.mean(xp.absolute(image_a - image_b))


def lhalf_error(image_a, image_b):
    # Backend:
    xp = Backend.get_xp_module(image_a)
    return xp.square(xp.mean(xp.sqrt(xp.absolute(image_a - image_b))))


def psnr(image_true, image_test):
    """
    Compute the peak signal to noise ratio (PSNR) for a [0,1] normalised image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    # Backend:
    xp = Backend.get_xp_module()

    # Move to backend:
    image_true = Backend.to_backend(image_true, dtype=xp.float32)
    image_test = Backend.to_backend(image_test, dtype=xp.float32)

    err = mean_squared_error(image_true, image_test)
    return 10 * xp.log10(1 / err)


def ssim(image_a, image_b, win_size=None):
    """
    Compute the mean structural similarity index between two images.

    Parameters
    ----------
    image_a, image_b : ndarray
        Images. Any dimensionality with same shape.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.


    Returns
    -------
    mssim : float
        The mean structural similarity index over the image.



    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`

    """

    # Backend:
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # Move to backend:
    image_a = Backend.to_backend(image_a, dtype=xp.float32)
    image_b = Backend.to_backend(image_b, dtype=xp.float32)

    if win_size is None:
        win_size = 7  # backwards compatibility

    if not (win_size % 2 == 1):
        raise ValueError("Window size must be odd.")

    # Constants:
    K1 = 0.01
    K2 = 0.03

    filter_func = sp.ndimage.uniform_filter
    filter_args = {"size": win_size}

    # filter has already normalized by NP
    cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(image_a, **filter_args)
    uy = filter_func(image_b, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(image_a * image_a, **filter_args)
    uyy = filter_func(image_b * image_b, **filter_args)
    uxy = filter_func(image_a * image_b, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = 1
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2, ux**2 + uy**2 + C1, vx + vy + C2)
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = _crop(S, pad).mean()

    return mssim


def _crop(ar, crop_width, copy=False, order="K"):
    """Crop array `ar` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),) or (before, after)`` specifies
        a fixed start and end crop for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.

    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """

    # Backend:
    xp = Backend.get_xp_module()

    ar = xp.array(ar, copy=False)

    if isinstance(crop_width, int):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], int):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f"crop_width has an invalid length: {len(crop_width)}\n"
                "crop_width should be a sequence of N pairs, "
                "a single pair, or a single integer"
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f"crop_width has an invalid length: {len(crop_width)}\n"
            "crop_width should be a sequence of N pairs, "
            "a single pair, or a single integer"
        )

    slices = tuple(slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops))
    if copy:
        cropped = xp.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped
