import itertools
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from arbol import aprint, asection
from scipy.optimize import minimize, shgo

from dexp.processing.denoising.metrics import mean_squared_error
from dexp.utils import dict_or, xpArray
from dexp.utils.backends import Backend


def calibrate_denoiser(
    image: xpArray,
    denoise_function: Callable[[Any], xpArray],
    denoise_parameters: Dict[str, List[Union[float, int]]],
    setup_function: Optional[Callable[[xpArray], Any]] = None,
    mode: str = "shgo+lbfgs",
    max_evaluations: int = 4096,
    stride: int = 4,
    loss_function: Callable = mean_squared_error,
    display: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates denoiser using self-supervised loss from Batson & Royer*
    Derived from code here:
    https://scikit-image.org/docs/dev/auto_examples/filters/plot_j_invariant_tutorial.html
    Reference: "Noise2Self: Blind Denoising by Self-Supervision, International
    Conference on Machine Learning, p. 524-533 (2019)"

    This 'classic_denoisers' version uses a 'brute-force' optimizer. Good when the
    denoiser is fast enough and the parameter space to explore small enough.

    Parameters
    ----------
    image: ArrayLike
        Image to calibate denoiser with.
    denoise_function: Callable
        Denosing function to calibrate. Should take an image as first parameter,
        all other parameters should have defaults
    denoise_parameters:
        Dictionary with keys corresponding to parameters of the denoising function.
        Values are either: (i) a list of possible values (categorical parameter),
        or (ii) a tuple of floats defining the bounds of that numerical parameter.
    setup_function : Callable, optional
        Function to pre process / setup denoising input.
    mode : str
        Optimisation mode. Can be: 'bruteforce', 'lbfgs' or 'shgo'.
    max_evaluations: int
        Maximum number of function evaluations during optimisation.
    stride: int
        Stride to compute self-supervised loss.
    loss_function: Callable
        Loss/Error function: takes two arrays and returns a distance-like function.
        Can be:  structural_error, mean_squared_error, _mean_absolute_error
    display_images: bool
        If True the denoised images for each parameter tested are displayed.
        this _will_ be slow.
    other_fixed_parameters: dict
        Other fixed parameters to pass to the denoiser function.


    Returns
    -------
    Dictionary with optimal parameters

    """
    # Move image to backend:
    image = Backend.to_backend(image)

    aprint(f"Calibrating denoiser on image of shape: {image.shape}")
    aprint(f"Stride for Noise2Self loss: {stride}")
    aprint(f"Fixed parameters: {other_fixed_parameters}")

    # Pass fixed parameters:
    denoise_function = partial(denoise_function, **other_fixed_parameters)

    with asection(f"Calibrating denoiser with method: {mode}"):
        best_parameters = _calibrate_denoiser_search(
            image,
            denoise_function,
            denoise_parameters=denoise_parameters,
            setup_function=setup_function,
            mode=mode,
            max_evaluations=max_evaluations,
            stride=stride,
            loss_function=loss_function,
            display_images=display,
        )

    aprint(f"Best parameters are: {best_parameters}")

    return dict_or(best_parameters, other_fixed_parameters)


def _interpolate_image(image: xpArray):

    # Backend:
    sp = Backend.get_sp_module(image)

    conv_filter = sp.ndimage.generate_binary_structure(image.ndim, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    interpolation = sp.ndimage.convolve(image, conv_filter, mode="mirror")

    return interpolation


def _generate_mask(image: xpArray, stride: int = 4):

    # Generate slice for mask:
    spatialdims = image.ndim
    n_masks = stride**spatialdims
    phases = np.unravel_index(n_masks // 2, (stride,) * len(image.shape[:spatialdims]))
    mask = tuple(slice(p, None, stride) for p in phases)

    return mask


def _product_from_dict(dictionary: Dict[str, List[Union[float, int]]]):
    """Utility function to convert parameter ranges to parameter combinations.

    Converts a dict of lists into a list of dicts whose values consist of the
    cartesian product of the values in the original dict.

    Parameters
    ----------
    dictionary : dict of lists
        Dictionary of lists to be multiplied.

    Yields
    ------
    selections : dicts of values
        Dicts containing individual combinations of the values in the input
        dict.
    """
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))


def _calibrate_denoiser_search(
    image: xpArray,
    denoise_function: Callable[[Any], xpArray],
    denoise_parameters: Dict[str, List[Union[float, int]]],
    setup_function: Optional[Callable[[xpArray], Any]],
    mode: str,
    max_evaluations: int,
    stride=4,
    loss_function: Callable = mean_squared_error,  # _structural_loss, #
    display_images: bool = False,
):
    """Return a parameter search history with losses for a denoise function.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : Callable
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    setup_function : Callable, optional
        Function to pre process / setup denoising input.
    mode : str
        Optimisation mode. Can be: "bruteforce", "lbfgs" or "shgo".
    max_evaluations: int
        Maximum number of function evaluations during optimisation.
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    loss_function : Callable
        Loss function to use
    display : bool
        When True the resulting images are displayed with napari

    Returns
    -------
    parameters_tested : list of dict
        List of parameters tested for `denoise_function`, as a dictionary of
        kwargs.
    losses : list of int
        Self-supervised loss for each set of parameters in `parameters_tested`.
    """

    # Move image to backend:
    image = Backend.to_backend(image)

    # Generate mask:
    mask = _generate_mask(image, stride)
    masked_image = image.copy()
    masked_image[mask] = _interpolate_image(image)[mask]

    # denoised images are kept here:
    denoised_images = []

    # Parameter names:
    parameter_names = list(denoise_parameters.keys())

    # Best parameters (to be found):
    best_parameters = None

    # Setting up denoising
    if setup_function is not None:
        denoising_input = setup_function(masked_image)
    else:
        denoising_input = masked_image

    # Function to optimise:
    def _loss_func(**_denoiser_kwargs):
        # We compute the J-inv loss:
        denoised = denoise_function(denoising_input, **_denoiser_kwargs)
        loss = loss_function(denoised[mask], image[mask])

        if math.isnan(loss) or math.isinf(loss):
            loss = math.inf

        aprint(f"J-inv loss is: {loss}")

        loss = Backend.to_numpy(loss)

        if display_images and not (math.isnan(loss) or math.isinf(loss)):
            denoised = denoise_function(image, **_denoiser_kwargs)
            denoised_images.append(denoised)

        return -float(loss)

    best_parameters = None

    if "bruteforce" in mode:
        with asection(f"Searching by brute-force for the best denoising parameters among: {denoise_parameters}"):

            num_rounds = 4
            best_loss = -math.inf
            for _ in range(num_rounds):
                # expand ranges:
                expanded_denoise_parameters = {n: np.arange(*r) for (n, r) in denoise_parameters.items()}
                # Generate all possible combinations:
                cartesian_product_of_parameters = list(_product_from_dict(expanded_denoise_parameters))
                for denoiser_kwargs in cartesian_product_of_parameters:
                    with asection(f"computing J-inv loss for: {denoiser_kwargs}"):
                        loss = _loss_func(**denoiser_kwargs)
                        if loss > best_loss:
                            best_loss = loss
                            best_parameters = denoiser_kwargs

    if "shgo" in mode:
        with asection(
            "Searching by 'simplicial homology global optimization' (SHGO)"
            f"the best denoising parameters among: {denoise_parameters}"
        ):
            if best_parameters is None:
                x0 = tuple(0.5 * (v[1] - v[0]) for v in denoise_parameters.values())
            else:
                x0 = tuple(best_parameters[k] for k in denoise_parameters.keys())

            bounds = list(v[0:2] for v in denoise_parameters.values())

            # Impedance mismatch:
            def _func(*_denoiser_kwargs):
                param_dict = {n: v for (n, v) in zip(parameter_names, tuple(_denoiser_kwargs[0]))}
                value = -_loss_func(**param_dict)
                return value

            result = shgo(_func, bounds, sampling_method="sobol", options={"maxev": max_evaluations})
            aprint(result)
            best_parameters = dict({n: v for (n, v) in zip(parameter_names, result.x)})

    if "lbfgs" in mode:
        with asection(f"Searching by 'Limited-memory BFGS' the best denoising parameters among: {denoise_parameters}"):

            if best_parameters is None:
                x0 = tuple(0.5 * (v[1] - v[0]) for v in denoise_parameters.values())
            else:
                x0 = tuple(best_parameters[k] for k in denoise_parameters.keys())

            bounds = list(v[0:2] for v in denoise_parameters.values())

            # Impedance mismatch:
            def _func(*_denoiser_kwargs):
                param_dict = {n: v for (n, v) in zip(parameter_names, tuple(_denoiser_kwargs[0]))}
                value = -_loss_func(**param_dict)
                return value

            result = minimize(
                fun=_func,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxfun=max_evaluations, eps=1e-2, ftol=1e-9, gtol=1e-9),
            )
            aprint(result)
            best_parameters = dict({n: v for (n, v) in zip(parameter_names, result.x)})

    if display_images:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(Backend.to_numpy(image), name="image")
        viewer.add_image(np.stack([Backend.to_numpy(i) for i in denoised_images]), name="denoised")
        napari.run()

    return best_parameters
