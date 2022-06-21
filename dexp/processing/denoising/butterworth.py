from functools import partial
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal._signaltools import _centered

from dexp.processing.crop.representative_crop import representative_crop
from dexp.processing.denoising.j_invariance import calibrate_denoiser
from dexp.utils import dict_or, xpArray
from dexp.utils.backends import Backend, CupyBackend
from dexp.utils.backends.cupy_backend import is_cupy_available

try:
    import cupyx

    rsqrt = cupyx.rsqrt

except ImportError:

    def rsqrt(x: xpArray) -> xpArray:
        return 1 / np.sqrt(x)


def calibrate_denoise_butterworth(
    image: xpArray,
    mode: str = "full",
    axes: Optional[Tuple[int, ...]] = None,
    padding: int = 32,
    min_freq: float = 0.001,
    max_freq: float = 1.0,
    num_freq: int = 32,
    min_order: float = 0.5,
    max_order: float = 6.0,
    num_order: int = 32,
    crop_size_in_voxels: Optional[int] = 256**3,
    display: bool = False,
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

    padding: int
        Amount of padding to be added to avoid edge effects.
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
    crop = representative_crop(image, crop_size=crop_size_in_voxels, display=False)

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
        dict(axes=axes),
    )

    if mode == "isotropic":
        # Partial function:
        _denoise_butterworth = partial(_apply_butterworth, **other_fixed_parameters, out_shape=crop.shape)

        # Parameters to test when calibrating the denoising algorithm
        parameter_ranges = {"freq_cutoff": freq_cutoff_range, "order": order_range}

    elif mode == "xy-z" and image.ndim == 3:
        # Partial function with parameter impedance match:
        def _denoise_butterworth(*args, **kwargs):
            freq_cutoff_xy = kwargs.pop("freq_cutoff_xy")
            freq_cutoff_z = kwargs.pop("freq_cutoff_z")
            _freq_cutoff = (freq_cutoff_xy, freq_cutoff_xy, freq_cutoff_z)
            return _apply_butterworth(
                *args,
                freq_cutoff=_freq_cutoff,
                out_shape=crop.shape,
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
            return _apply_butterworth(
                *args,
                out_shape=crop.shape,
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
            setup_function=partial(_setup_butterworth_denoiser, axes=axes, padding=padding),
            mode="shgo+lbfgs",
            max_evaluations=1000,
            display=display,
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
        freq_cutoff = (freq_cutoff_z, freq_cutoff_xy, freq_cutoff_xy)
        best_parameters = dict_or(best_parameters, {"freq_cutoff": freq_cutoff})

    if isinstance(Backend.current(), CupyBackend):
        # Free plane cache to avoid running out of memory
        from cupy.fft.config import get_plan_cache

        get_plan_cache().clear()

    return denoise_butterworth, best_parameters


def denoise_butterworth(
    image,
    axes: Optional[Tuple[int, ...]] = None,
    freq_cutoff: Union[float, Sequence[float]] = 0.5,
    order: float = 1,
    padding: int = 32,
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

    padding: int
        Amount of padding to be added to avoid edge effects.

    Returns
    -------
    Denoised image

    """
    if not isinstance(freq_cutoff, Iterable):
        freq_cutoff = tuple((freq_cutoff,) * image.ndim)

    # Computes fft and grid distance map of input image
    data = _setup_butterworth_denoiser(image, axes=axes, padding=padding)
    return _apply_butterworth(data, axes, freq_cutoff, order, image.shape)


def _apply_butterworth(
    data: Tuple[xpArray, Sequence[xpArray]],
    axes: Optional[Sequence[int]],
    freq_cutoff: Tuple[float],
    order: float,
    out_shape: Tuple[int],
) -> xpArray:
    """
    Applies the butterworth filter to a pre computed image in the freq. domain
    and a grid for each axes for distance map computation.

    Parameters
    ----------
    data : Tuple[xpArray, Sequence[xpArray]]
        Image in the freq. domain and a grid for each axes.
    axes : Optional[Sequence[int]]
        Axes selected for filtering
    freq_cutoff : Tuple[float]
        Freq. cutoff parameter.
    order : float
        Butterworth order parameter.
    out_shape : Tuple[int]
        Original image input shape, used to crop the data after the inverse FFT.

    Returns
    -------
    xpArray
        Butterworth filtered image.
    """

    image_f, grid = data

    dist = np.zeros_like(image_f, dtype=np.float32)
    for axis, fc in zip(grid, freq_cutoff):
        dist += np.square(axis / fc)

    # Apply filter:
    image_f = _butterworth_filter(image_f, dist, order)

    # Shift back:
    image_f = np.fft.ifftshift(image_f, axes=axes)

    # Back in real space:
    denoised = np.real(np.fft.ifftn(image_f, axes=axes))

    # Crop to remove padding:
    denoised = _centered(denoised, out_shape)

    return denoised


def _setup_butterworth_denoiser(
    image: xpArray, axes: Optional[Tuple[int, ...]], padding: int
) -> Tuple[xpArray, xpArray]:
    """Pre computes the butterworth forward step.

    Parameters
    ----------
    image : xpArray
        Input image.

    axes: Optional[Tuple[int,...]]
        Axes over which to apply lowpass filtering.

    padding: int
        Padding to be added to avoid edge effects.

    Returns
    -------
    Tuple[xpArray, xpArray]
        Input image in the freq. domain and the distance map.
    """
    xp = Backend.get_xp_module(image)

    # Convert image to float if needed:
    image = image.astype(dtype=np.float32)

    # Default axes:
    if axes is None:
        axes = tuple(range(image.ndim))

    # Selected axes:
    selected_axes = tuple((a in axes) for a in range(image.ndim))

    # First we need to pad the image.
    # By how much? this depends on how much low filtering we need to do:
    pad_width = tuple((padding, padding) if selected else 0 for selected in selected_axes)

    # pad image:
    image = np.pad(image, pad_width=pad_width, mode="reflect")

    # Move to frequency space:
    image_f = np.fft.fftn(image, axes=axes)

    # Center frequencies:
    image_f = np.fft.fftshift(image_f, axes=axes)

    # Computes grid for squared distance
    axis_grid = tuple((xp.linspace(-1, 1, s) if sa else xp.zeros((s,))) for sa, s in zip(selected_axes, image.shape))
    f = xp.meshgrid(*axis_grid, indexing="ij")

    return image_f, f


def _butterworth_filter(image_f: xpArray, f: xpArray, order: float) -> xpArray:
    if isinstance(image_f, np.ndarray):
        return image_f / np.sqrt(1 + f**order)
    else:
        # Faster operation if cupy array is used
        return image_f * rsqrt(1 + f**order)


# "Compiles" to cupy if available
if is_cupy_available():
    import cupy

    _butterworth_filter = cupy.fuse(_butterworth_filter)
