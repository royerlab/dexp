from arbol.arbol import asection

from dexp.utils.backends import Backend


@asection("Generating synthetic binary blobs data")
def binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.5, rng=42):
    """
    Generate synthetic binary image with several rounded blob-like objects.

    Parameters
    ----------
    length : int, optional
        Linear size of output image.
    blob_size_fraction : float, optional
        Typical linear size of blob, as a fraction of ``length``, should be
        smaller than 1.
    n_dim : int, optional
        Number of dimensions of output image.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs (where the output is 1).
        Should be in [0, 1].
    rng : int or Generator
        If an integer is provided it's used to create an numpy (cupy) default random number generator.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image

    Examples
    --------
    >>> data.binary_blobs(length=5, blob_size_fraction=0.2, seed=1)
    array([[ True, False,  True,  True,  True],
           [ True,  True,  True, False,  True],
           [False,  True, False,  True,  True],
           [ True, False, False,  True,  True],
           [ True, False, False, False,  True]])
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.1)
    >>> # Finer structures
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.05)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = data.binary_blobs(length=256, volume_fraction=0.3)

    Notes:
    ------
    Code adapted from scikit-image
    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if isinstance(rng, int):
        rng = xp.random.RandomState(seed=rng)

    shape = tuple([length] * n_dim)
    mask = xp.zeros(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)
    points = (length * rng.rand(n_dim, n_pts)).astype(int)
    mask[tuple(indices for indices in points)] = 1

    mask = sp.ndimage.gaussian_filter(mask, sigma=0.25 * length * blob_size_fraction)
    threshold = xp.percentile(mask, 100 * (1 - volume_fraction))
    return xp.logical_not(mask < threshold)
