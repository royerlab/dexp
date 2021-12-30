def nd_split_slices(array_shape, chunks, margins=None):
    """
    nd_split_slices

    Parameters
    ----------
    array_shape : tuple
    nb_slices : list
    do_shuffle : tuple
    margins : tuple

    Returns
    -------

    """
    if not array_shape:
        yield ()
        return

    if margins is None:
        margins = (0,) * len(array_shape)

    dim_width = array_shape[-1]

    for outer in nd_split_slices(array_shape[:-1], chunks[:-1], margins=margins[:-1]):

        slice_width = chunks[-1]
        slice_margin = margins[-1]

        slice_start_range = list(range(0, dim_width, slice_width))

        for slice_start in slice_start_range:
            start = max(0, slice_start - slice_margin)
            stop = min(slice_start + slice_width + slice_margin, dim_width)
            yield outer + (slice(start, stop, 1),)


def remove_margin_slice(array_shape, slice_with_margin, slice_without_margin):
    """
    Remove_margin_slice

    Parameters
    ----------
    array_shape : tuple
    slice_with_margin : array_like
    slice_without_margin : array_like

    Returns
    -------
    sliced_tuple : tuple

    """
    slice_tuple = tuple(
        slice(max(0, v.start - u.start), min(v.stop - u.start, l), 1)
        for l, u, v in zip(array_shape, slice_with_margin, slice_without_margin)
    )
    return slice_tuple
