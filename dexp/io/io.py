import numpy
from tifffile import imsave


def tiff_save(file, img, axes='ZYX', compress=0, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """

    # convert to imagej-compatible data type
    t = img.dtype
    if 'float' in t.name:
        t_new = numpy.float32
    elif 'uint' in t.name:
        t_new = numpy.uint16 if t.itemsize >= 2 else numpy.uint8
    elif 'int' in t.name:
        t_new = numpy.int16
    else:
        t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        print("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, numpy.dtype(t_new)))

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs, compress=compress, metadata={'axes': axes})  # ,
