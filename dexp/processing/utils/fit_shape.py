from dexp.processing.backends.backend import Backend


def fit_shape(backend: Backend, array, shape):
    length_diff = tuple(u - v for u, v in zip(shape, array.shape))

    if any(x < 0 for x in length_diff):
        # we need to crop at least one dimension:
        slicing = tuple(slice(0, s) for s in shape)
        array = array[slicing]

    # Independently of whether we had to crop a dimension, we proceed with eventual padding:
    length_diff = tuple(u - v for u, v in zip(shape, array.shape))

    if any(x > 0 for x in length_diff):
        xp = backend.get_xp_module()
        pad_width = tuple(tuple((0, d)) for d in length_diff)
        array = xp.pad(array, pad_width=pad_width)

    return array
