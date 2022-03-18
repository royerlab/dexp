from functools import partial
from typing import Optional, Sequence, Tuple, Union

from numba import jit

from dexp.processing.crop.representative_crop import representative_crop
from dexp.processing.denoising.j_invariance import calibrate_denoiser
from dexp.utils import dict_or, xpArray
from dexp.utils.backends import Backend, CupyBackend


def calibrate_denoise_butterworth(
    image: xpArray,
    mode: str = "full",
    axes: Optional[Tuple[int, ...]] = None,
    max_padding: int = 32,
    min_freq: float = 0.001,
    max_freq: float = 1.0,
    num_freq: int = 32,
    min_order: float = 0.5,
    max_order: float = 6.0,
    num_order: int = 32,
    crop_size_in_voxels: Optional[int] = 1280000,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates the Butterworth denoiser for the given image and returns the optimal
    parameters obtained using the N2S loss.

    Parameters
    ----------
    image: ArrayLike
        Image to calibrate Sobolev denoiser for.

    mode: str
        Possible modes are: 'isotropic' for isotropic meaning only one
        frequency cut-off is calibrated for all axes , 'xy-z' for 3D stacks where
        the cut-off frequency for the x and y axes is the same but different for
        the z axis, and 'full' for which all frequency cut-offs are different.

    axes: Optional[Tuple[int,...]]
        Axes over which to apply low-pass filtering.
        (advanced)

    max_padding: int
        Maximum amount of padding to be added to avoid edge effects.
        (advanced)

    min_freq: float
        Minimum cutoff frequency to use for calibration. Must be within [0,1],
        typically close to zero.
        (advanced)

    max_freq: float
        Maximum cutoff frequency to use for calibration. Must be within [0,1],
        typically close to one.
        (advanced)

    num_freq: int
        Number of frequencies to use for calibration.

    min_order: float
        Minimal order for the Butterworth filter to use for calibration.
        (advanced)

    max_order: float
        Maximal order for the Butterworth filter to use for calibration.
        (advanced)

    num_order: int
        Number of orders to use for calibration.

    crop_size_in_voxels: int or None for default
        Number of voxels for crop used to calibrate denoiser.
        (advanced)

    display_images: bool
        When True the denoised images encountered during optimisation are shown.
        (advanced)

    other_fixed_parameters: dict
        Any other fixed parameters. (advanced)

    Returns
    -------
    Denoising function, dictionary containing optimal parameters,
    and free memory needed in bytes for computation.
    """

    # Backend:
    xp = Backend.get_xp_module(image)

    # Convert image to float if needed:
    image = image.astype(dtype=xp.float32, copy=False)

    # obtain representative crop, to speed things up...
    crop = representative_crop(image, crop_size=crop_size_in_voxels)

    # ranges:
    freq_cutoff_range = (
        min_freq,
        max_freq,
        (max_freq - min_freq) / num_freq,
    )  # numpy.arange(min_freq, max_freq, (max_freq-min_freq)/num_freq)
    order_range = (
        min_order,
        max_order,
        (max_order - min_order) / num_order,
    )  # numpy.arange(min_order, max_order, (max_order-min_order)/num_order)

    # Combine fixed parameters:
    other_fixed_parameters = dict_or(
        other_fixed_parameters,
        {"max_padding": max_padding, "axes": axes},
    )

    if mode == "isotropic":
        # Partial function:
        _denoise_butterworth = partial(denoise_butterworth, **other_fixed_parameters)

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {"freq_cutoff": freq_cutoff_range, "order": order_range}

    elif mode == "xy-z" and image.ndim == 3:
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            freq_cutoff_xy = kwargs.pop("freq_cutoff_xy")
            freq_cutoff_z = kwargs.pop("freq_cutoff_z")
            _freq_cutoff = (freq_cutoff_xy, freq_cutoff_xy, freq_cutoff_z)
            return denoise_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                **dict_or(kwargs, other_fixed_parameters),
            )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {
            "freq_cutoff_xy": freq_cutoff_range,
            "freq_cutoff_z": freq_cutoff_range,
            "order": order_range,
        }

    elif mode == "full":
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            _freq_cutoff = tuple(kwargs.pop(f"freq_cutoff_{i}") for i in range(image.ndim))
            return denoise_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                **dict_or(kwargs, other_fixed_parameters),
            )

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {f"freq_cutoff_{i}": freq_cutoff_range for i in range(image.ndim)}
        parameter_ranges["order"] = order_range

    else:
        raise ValueError(f"Unsupported denoising mode: {mode}")

    # Calibrate denoiser
    best_parameters = dict_or(
        calibrate_denoiser(
            crop,
            _denoise_butterworth,
            denoise_parameters=parameter_ranges,
            mode="lbfgs",  # ,"shgo"
            display_images=display_images,
        ),
        other_fixed_parameters,
    )

    if mode == "full":
        # We need to adjust a bit the type of parameters passed to the denoising function:
        freq_cutoff = tuple(best_parameters.pop(f"freq_cutoff_{i}") for i in range(image.ndim))
        best_parameters = dict_or(best_parameters, {"freq_cutoff": freq_cutoff})
    elif mode == "xy-z":
        # We need to adjust a bit the type of parameters passed to the denoising function:
        freq_cutoff_xy = best_parameters.pop("freq_cutoff_xy")
        freq_cutoff_z = best_parameters.pop("freq_cutoff_z")
        freq_cutoff = (freq_cutoff_xy, freq_cutoff_xy, freq_cutoff_z)
        best_parameters = dict_or(best_parameters, {"freq_cutoff": freq_cutoff})

    return denoise_butterworth, best_parameters


def denoise_butterworth(
    image,
    axes: Optional[Tuple[int, ...]] = None,
    freq_cutoff: Union[float, Sequence[float]] = 0.5,
    order: float = 1,
    max_padding: int = 32,
):
    """
    Denoises the given image by applying a configurable <a
    href="https://en.wikipedia.org/wiki/Butterworth_filter">Butterworth
    lowpass filter</a>. Remarkably good when your signal
    does not have high-frequencies beyond a certain cutoff frequency.
    This is probably the first algorithm that should be tried of all
    currently available in Aydin. It is actually quite impressive how
    well this performs in practice. If the signal in your images is
    band-limited as is often the case for microscopy images, this
    denoiser will work great.
    \n\n
    Note: We recommend applying a variance stabilisation transform
    to improve results for images with non-Gaussian noise.

    Parameters
    ----------
    image: ArrayLike
        Image to be denoised

    axes: Optional[Tuple[int,...]]
        Axes over which to apply lowpass filtering.

    freq_cutoff: Union[float, Sequence[float]]
        Single or sequence cutoff frequency, must be within [0, 1]

    order: float
        Filter order, typically an integer above 1.

    max_padding: int
        Maximum amount of padding to be added to avoid edge effects.



    Returns
    -------
    Denoised image

    """

    # Backend:
    xp = Backend.get_xp_module(image)
    sp = Backend.get_sp_module(image)

    # Convert image to float if needed:
    image = image.astype(dtype=xp.float32, copy=False)

    # Normalise freq_cutoff argument to tuple:
    if type(freq_cutoff) is not tuple:
        freq_cutoff = tuple((freq_cutoff,) * image.ndim)

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # Selected axes:
    selected_axes = tuple((a in axes) for a in range(image.ndim))

    # First we need to pad the image.
    # By how much? this depends on how much low filtering we need to do:
    pad_width = tuple(
        ((_apw(fc, max_padding), _apw(fc, max_padding)) if sa else (0, 0)) for sa, fc in zip(selected_axes, freq_cutoff)
    )

    # pad image:
    image = xp.pad(image, pad_width=pad_width, mode="reflect")

    # Move to frequency space:
    image_f = _fftn(axes, image, sp)

    # Center frequencies:
    image_f = sp.fft.fftshift(image_f, axes=axes)

    # Compute squared distance image:
    f = _compute_distance_image(freq_cutoff, image, selected_axes)

    # Chose filter implementation:
    if isinstance(Backend.current(), CupyBackend):
        import cupy

        filter = cupy.vectorize(_filter)
    else:
        filter = jit(nopython=True, parallel=True)(_filter)

    # Apply filter:
    image_f = filter(image_f, f, order)

    # Shift back:
    image_f = sp.fft.ifftshift(image_f, axes=axes)

    # Back in real space:
    denoised = xp.real(_ifftn(axes, image_f, sp))

    # Crop to remove padding:
    denoised = denoised[tuple(slice(u, -v) for u, v in pad_width)]

    return denoised


def _ifftn(axes, image_f, sp):
    try:
        return sp.fft.ifftn(image_f, axes=axes, workers=-1)
    except TypeError:
        # Some backends do not support the workers argument:
        return sp.fft.ifftn(image_f, axes=axes)


def _fftn(axes, image, sp):
    try:
        return sp.fft.fftn(image, axes=axes, workers=-1)
    except TypeError:
        # Some backends do not support the workers argument:
        return sp.fft.fftn(image, axes=axes)


def _compute_distance_image(freq_cutoff, image, selected_axes):

    # Backend:
    xp = Backend.get_xp_module(image)

    f = xp.zeros_like(image, dtype=xp.float32)
    axis_grid = tuple((xp.linspace(-1, 1, s) if sa else xp.zeros((s,))) for sa, s in zip(selected_axes, image.shape))
    for fc, x in zip(freq_cutoff, xp.meshgrid(*axis_grid, indexing="ij")):
        f += (x / fc) ** 2
    return f


def _apw(freq_cutoff, max_padding):
    return min(max_padding, max(1, int(1.0 / (1e-10 + freq_cutoff))))


def _filter(image_f, f, order):

    image_f *= (1 + f ** order) ** (-0.5)
    return image_f
