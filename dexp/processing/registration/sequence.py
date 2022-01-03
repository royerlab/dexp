from typing import List, Optional, Tuple

import dask
import numpy
from arbol import aprint, asection
from dask.array import Array
from joblib import Parallel, delayed

from dexp.processing.registration.model.sequence_registration_model import (
    SequenceRegistrationModel,
)
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration.translation_nd_proj import (
    register_translation_proj_nd,
)
from dexp.processing.utils.center_of_mass import center_of_mass
from dexp.processing.utils.linear_solver import linsolve
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def image_stabilisation(
    image: xpArray,
    axis: int,
    preload_images: bool = True,
    mode: str = "translation",
    max_range: int = 7,
    min_confidence: float = 0.5,
    enable_com: bool = False,
    quantile: float = 0.5,
    bounding_box: bool = False,
    tolerance: float = 1e-7,
    order_error: float = 2.0,
    order_reg: float = 1.0,
    alpha_reg: float = 0.1,
    detrend: bool = False,
    debug_output: str = None,
    workers: int = 1,
    internal_dtype=None,
    **kwargs,
) -> SequenceRegistrationModel:
    """
    Computes a sequence stabilisation model for an image sequence.

    Parameters
    ----------
    image: image to stabilise.
    axis: sequence axis along which to stabilise image.
    preload_images: boolean indicating to preload data or not.
    mode: registration mode. For now only 'translation' is available.
    max_range: maximal distance, in time points, between pairs of images to registrate.
    min_confidence: minimal confidence to accept a pairwise registration
    enable_com: enable center of mass fallback when standard registration fails.
    quantile: quantile to cut-off background in center-of-mass calculation
    bounding_box: if True, the center of mass of the bounding box of non-zero pixels is returned.
    tolerance: tolerance for linear solver.
    order_error: order for linear solver error term.
    order_reg: order for linear solver regularisation term.
    alpha_reg: multiplicative coefficient for regularisation term.
    detrend: removes linear detrend from stabilized image.
    internal_dtype : internal dtype for computation
    **kwargs: argument passthrough to the pairwise registration method, see 'register_translation_nd'.

    Returns
    -------
    Sequence registration model

    """
    assert 0 <= axis < image.ndim
    length = image.shape[axis]

    xp = Backend.get_xp_module()

    image_sequence = None
    if preload_images:
        with asection("Preloading images to backend..."):
            image_sequence = list(
                Backend.to_backend(Backend.get_xp_module(image).take(image, i, axis=axis)) for i in range(length)
            )

    scales = list(i for i in range(max_range) if i < length)

    aprint(f"Scales: {scales}")

    with asection(f"Registering image sequence of length: {length}"):
        ndim = image.ndim - 1

        uv_set = set()
        with asection("Enumerating pairwise registrations needed..."):

            for scale_index, scale in enumerate(scales):
                for offset in range(0, scale):
                    for u in range(offset, length, scale):
                        v = u + scale

                        # if u < 0:
                        #     u = 0
                        if v >= length:
                            continue

                        atuple = tuple((u, v))
                        # aprint(f"Pair: ({u}, {v}) added")
                        uv_set.add(atuple)

        with asection(f"Computing pairwise registrations for {len(uv_set)} (u,v) pairs..."):

            def _compute_model(pair: Tuple[int, int]) -> Optional[TranslationRegistrationModel]:
                u, v = pair
                if image_sequence:
                    image_u = image_sequence[u]
                    image_v = image_sequence[v]
                elif isinstance(image, Array):
                    image_u = dask.array.take(image, u, axis=axis)
                    image_v = dask.array.take(image, v, axis=axis)
                else:
                    image_u = xp.take(image, u, axis=axis)
                    image_v = xp.take(image, v, axis=axis)
                model = _pairwise_registration(
                    u,
                    v,
                    image_u,
                    image_v,
                    mode,
                    min_confidence,
                    enable_com,
                    quantile,
                    bounding_box,
                    internal_dtype,
                    **kwargs,
                )
                return model

            pairwise_models = Parallel(n_jobs=workers)(delayed(_compute_model)(pair) for pair in uv_set)
            pairwise_models = [model for model in pairwise_models if model is not None]

        nb_models = len(pairwise_models)
        aprint(f"Number of models obtained: {nb_models} for a sequence of length:{length}")

        if debug_output is not None:
            with asection(f"Generating pairwise registration confidence matrix: '{debug_output}' "):
                import matplotlib.pyplot as plt
                import seaborn as sns

                sns.set_theme()
                plt.clf()
                plt.cla()

                array = numpy.zeros((length, length))
                for tp, model in enumerate(pairwise_models):
                    u = model.u
                    v = model.v
                    array[u, v] = model.overall_confidence()
                sns.heatmap(array)

                plt.savefig(debug_output + "_prcm.pdf")

        with asection("Solving for optimal sequence registration"):
            with NumpyBackend():
                xp = Backend.get_xp_module()
                sp = Backend.get_sp_module()

                if mode == "translation":

                    # prepares list of models:
                    translation_models: List[TranslationRegistrationModel] = list(
                        TranslationRegistrationModel(xp.zeros((ndim,), dtype=internal_dtype)) for _ in range(length)
                    )

                    # initialise count for average:
                    for model in translation_models:
                        model.count = 0

                    for d in range(ndim):

                        # instantiates system to solve with zeros:
                        a = xp.zeros((nb_models + 1, length), dtype=xp.float32)
                        y = xp.zeros((nb_models + 1,), dtype=xp.float32)

                        # iterates through all pairwise registrations:
                        zero_vector = None
                        for tp, model in enumerate(pairwise_models):
                            u = model.u
                            v = model.v
                            confidence = model.overall_confidence()

                            if tp == 0:
                                # we make sure that all shifts are relative to the first timepoint:
                                zero_vector = model.shift_vector[d].copy()

                            vector = model.shift_vector[d] - zero_vector

                            # Each pairwise registration defines a constraint that is added to the matrix:
                            y[tp] = vector.copy()
                            a[tp, u] = +1
                            a[tp, v] = -1

                            # For each time point we collect the average confidence of all the pairwise_registrations:
                            translation_models[u].confidence += confidence
                            translation_models[u].count += 1
                            translation_models[v].confidence += confidence
                            translation_models[v].count += 1

                        # Forces solution to have no displacement for first time point:
                        y[-1] = 0
                        a[-1, 0] = 1

                        # move matrix to sparse domain (much faster!):
                        a = sp.sparse.coo_matrix(a)

                        # solve system:
                        x_opt = linsolve(
                            a, y, tolerance=tolerance, order_error=order_error, order_reg=order_reg, alpha_reg=alpha_reg
                        )

                        # detrend:
                        if detrend:
                            x_opt = sp.signal.detrend(x_opt)

                        # sets the shift vectors for the resulting sequence reg model, and compute average confidences:
                        for tp in range(length):
                            translation_models[tp].shift_vector[d] = -x_opt[tp]
                            translation_models[tp].confidence /= translation_models[tp].count

                    model = SequenceRegistrationModel(model_list=translation_models)

                else:
                    raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")

            if debug_output is not None:
                with asection("Generating shift vector plots"):
                    model.plot(debug_output)

            return model

            # if not image_a.dtype == image_b.dtype:
            #     raise ValueError("Arrays must have the same dtype")
            #
            # if internal_dtype is None:
            #     internal_dtype = image_a.dtype
            #
            # if type(Backend.current()) is NumpyBackend:
            #     internal_dtype = xp.float32
            #
            # image_a = Backend.to_backend(image_a, dtype=internal_dtype)
            # image_b = Backend.to_backend(image_b, dtype=internal_dtype)


def _pairwise_registration(
    u, v, image_u, image_v, mode, min_confidence, enable_com, quantile, bounding_box, internal_dtype, **kwargs
):
    image_u = Backend.to_backend(image_u, dtype=internal_dtype)
    image_v = Backend.to_backend(image_v, dtype=internal_dtype)

    if mode == "translation":
        model = register_translation_proj_nd(image_u, image_v, _display_phase_correlation=False, **kwargs)
        model.u = u
        model.v = v
        confidence = model.overall_confidence()

        if confidence < min_confidence:
            if enable_com:
                offset_mode = f"p={quantile * 100}"
                com_u = center_of_mass(
                    image_u, mode="full", projection_type="max-min", offset_mode=offset_mode, bounding_box=bounding_box
                )
                com_v = center_of_mass(
                    image_v, mode="full", projection_type="max-min", offset_mode=offset_mode, bounding_box=bounding_box
                )
                model = TranslationRegistrationModel(shift_vector=com_u - com_v, confidence=min_confidence)
                model.u = u
                model.v = v
            else:
                model = None

    else:
        raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")

    return model
