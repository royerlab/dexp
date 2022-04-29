import numpy as np
from tifffile import imsave


def tiff_save(file, img, axes="ZYX", compress=0, **imsave_kwargs):
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
    img = np.asarray(img)

    t = img.dtype
    if "float" in t.name:
        t_new = np.float32
    elif "uint" in t.name:
        t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif "int" in t.name:
        t_new = np.int16
    else:
        t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        print(f"Converting data type from '{t}' to ImageJ-compatible '{np.dtype(t_new)}'.")

    imsave_kwargs["imagej"] = True
    imsave(file, img, **imsave_kwargs, compress=compress, metadata={"axes": axes})  # ,
