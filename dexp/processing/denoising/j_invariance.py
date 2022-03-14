import itertools
import math
from functools import partial
from typing import Callable, List, Union

import numpy
from arbol import aprint, asection
from dexp.utils.bayes_opt import BayesianOptimization, \
    SequentialDomainReductionTransformer

from dexp.processing.denoising.metrics import mean_squared_error
from dexp.utils import xpArray
from dexp.utils.backends import Backend




def calibrate_denoiser(
    image: xpArray,
    denoise_function: Callable,
    denoise_parameters: dict[List[Union[float, int]]],
    mode: str = "bayesian",
    stride: int = 4,
    loss_function: Callable=mean_squared_error,
    # _structural_loss, mean_squared_error, mean_absolute_error #
    display_images: bool = False,
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
    mode : str
        Optimisation mode. Can be: 'bruteforce' or 'bayesian'.
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

    best_parameters = _calibrate_denoiser_search(
        image,
        denoise_function,
        denoise_parameters=denoise_parameters,
        mode=mode,
        stride=stride,
        loss_function=loss_function,
        display_images=display_images,
    )

    aprint(f"Best parameters are: {best_parameters}")

    return best_parameters | other_fixed_parameters




def _j_invariant_loss(
    image: xpArray,
    denoise_function: Callable,
    mask: xpArray,
    loss_function: Callable = mean_squared_error,  # _structural_loss, #
    denoiser_kwargs=None,
):
    image = image.astype(dtype=numpy.float32, copy=False)

    denoised = _invariant_denoise(
        image,
        denoise_function=denoise_function,
        mask=mask,
        denoiser_kwargs=denoiser_kwargs,
    )

    loss = loss_function(image[mask], denoised[mask])

    return loss


def _invariant_denoise(
        image: xpArray,
        denoise_function: Callable,
        mask: xpArray,
        denoiser_kwargs=None):

    # Backend:
    xp = Backend.get_xp_module(image)

    image = image.astype(dtype=numpy.float32, copy=False)

    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    interp = _interpolate_image(image)
    output = xp.zeros_like(image)

    input_image = image.copy()
    input_image[mask] = interp[mask]
    output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]

    return output


def _interpolate_image(image: xpArray):

    # Backend:
    sp = Backend.get_sp_module(image)

    conv_filter = sp.ndimage.generate_binary_structure(image.ndim, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()

    interpolation = sp.ndimage.convolve(image, conv_filter, mode='mirror')

    return interpolation


def _generate_mask(image: xpArray,
                   stride: int = 4):

    # Generate slice for mask:
    spatialdims = image.ndim
    n_masks = stride ** spatialdims
    mask = _generate_grid_slice(
        image.shape[:spatialdims], offset=n_masks // 2, stride=stride
    )

    return mask


def _generate_grid_slice(shape: tuple[int,...],
                         offset: int,
                         stride: int = 3):
    phases = numpy.unravel_index(offset, (stride,) * len(shape))
    mask = tuple(slice(p, None, stride) for p in phases)
    return mask


def _mid_point(numerical_parameters_bounds):
    mid_point = tuple(0.5 * (u + v) for u, v in numerical_parameters_bounds)
    mid_point = numpy.array(mid_point)
    return mid_point


def _product_from_dict(dictionary: dict[List[Union[float, int]]]):
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
    denoise_function: Callable,
    denoise_parameters: dict[List[Union[float, int]]],
    mode: str,
    stride=4,
    loss_function: Callable = mean_squared_error,  # _structural_loss, #
    display_images: bool = False,
):
    """Return a parameter search history with losses for a denoise function.

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using `img_as_float`).
    denoise_function : function
        Denoising function to be calibrated.
    denoise_parameters : dict of list
        Ranges of parameters for `denoise_function` to be calibrated over.
    mode : str
        Optimisation mode. Can be: bruteforce or bayesian
    stride : int, optional
        Stride used in masking procedure that converts `denoise_function`
        to J-invariance.
    loss_function : Callable
        Loss function to use
    display_images : bool
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

    # denoised images are kept here:
    denoised_images = []

    # Best parameters (to be found):
    best_parameters = None

    # Function to optimise:
    def _function(**_denoiser_kwargs):
        # We compute the J-inv loss:
        loss = _j_invariant_loss(
            image,
            denoise_function,
            mask=mask,
            loss_function=loss_function,
            denoiser_kwargs=_denoiser_kwargs,
        )

        if math.isnan(loss) or math.isinf(loss):
            loss = math.inf
        aprint(f"J-inv loss is: {loss}")

        loss = Backend.to_numpy(loss)

        if display_images and not (math.isnan(loss) or math.isinf(loss)):
            denoised = denoise_function(image, **denoiser_kwargs)
            denoised_images.append(denoised)

        return -float(loss)

    if mode == "bruteforce":
        with asection(
            f"Searching by brute-force for the best denoising parameters among: {denoise_parameters}"
        ):
            # expand ranges:
            denoise_parameters = {n: numpy.arange(*r) for (n, r) in denoise_parameters.items()}

            # Generate all possible combinations:
            parameters_tested = list(_product_from_dict(denoise_parameters))

            best_loss = -math.inf
            for denoiser_kwargs in parameters_tested:
                with asection(f"computing J-inv loss for: {denoiser_kwargs}"):
                    loss = _function(**denoiser_kwargs)
                    if loss > best_loss:
                        best_loss = loss
                        best_parameters = denoiser_kwargs

    elif mode == "bayesian":

        # Bounded region of parameter space
        # expand ranges:
        denoise_parameters = {n: r[0:2] for (n, r) in
                              denoise_parameters.items()}

        bounds_transformer = SequentialDomainReductionTransformer(gamma_osc=0.7,
                                                                  gamma_pan=1.0,
                                                                  eta=0.99)

        optimizer = BayesianOptimization(
            f=_function,
            pbounds=denoise_parameters,
            verbose=2,
            # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
            bounds_transformer=bounds_transformer
        )

        optimizer.maximize(
            init_points=3**len(denoise_parameters),
            n_iter=500,
        )

        best_parameters = optimizer.max['params']

    if display_images:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(Backend.to_numpy(image), name='image')
        viewer.add_image(numpy.stack([Backend.to_numpy(i) for i in denoised_images]), name='denoised')
        napari.run()

    return best_parameters



