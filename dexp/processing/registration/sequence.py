import os
from math import log
from typing import List, Optional, Sequence

from arbol import aprint, asection
from dask.array import Array
from joblib import Parallel, delayed

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.translation_nd_proj import register_translation_maxproj_nd
from dexp.processing.utils.center_of_mass import center_of_mass
from dexp.processing.utils.linear_solver import linsolve


def image_stabilisation(image: 'Array',
                        axis: int = 0,
                        internal_dtype=None,
                        **kwargs
                        ) -> SequenceRegistrationModel:
    """
    Computes a sequence stabilisation model for an image sequence indexed along a specified axis.


    Parameters
    ----------
    image: image to stabilise
    axis: sequence axis along which to stabilise image
    internal_dtype : internal dtype for computation
    **kwargs: argument passthrough to the pairwise registration method.

    Returns
    -------
    Sequence registration model

    """
    xp = Backend.get_xp_module()

    image = Backend.to_backend(image, dtype=internal_dtype)

    length = image.shape[axis]
    image_sequence = list(xp.take(image, axis=axis, indices=range(0, length)))

    return image_sequence_stabilisation(image_sequence=image_sequence,
                                        **kwargs)


def image_sequence_stabilisation(image_sequence: Sequence['Array'],
                                 mode: str = 'translation',
                                 min_scale: Optional[int] = 0,
                                 max_scale: Optional[int] = None,
                                 min_confidence: float = 0.7,
                                 full_reg_scale: int = None,
                                 order_error: float = 1.0,
                                 order_reg: float = 2.0,
                                 alpha_reg: float = 1e-6,
                                 workers: int = -1,
                                 workersbackend: str = 'threading',
                                 internal_dtype=None,
                                 **kwargs
                                 ) -> SequenceRegistrationModel:
    """
    Computes a sequence stabilisation model for an image sequence.


    Parameters
    ----------
    image: image to stabilise
    axis: sequence axis along which to stabilise image
    mode: registration mode. For now only 'translation' is available.
    min_scale: Minimal scale to use for pairwise registration. Each scale corresponds roughly to a power of the golden ratio g**scale,
    max_scale: Maximal scale to use for pairwise registration. Each scale corresponds roughly to a power of the golden ratio g**scale,
    but to avoid hitting the same images too often, we snap these integers to the closest prime number (see magic numbers table in code). Best left to the maximum default
    (None) unless there is a very good reason not to.
    min_confidence: minimal confidence to accept a pairwise registration
    full_reg_scale: Scale below which a full registration is performed, above that only center of mass calculation is performed...
    solver_order: order for linear solver. We use the same order for both error and regularisation terms.
    workers: Number of worker threads to spawn, if -1 then number of workers = number cpu cores
    workers_backend: What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    internal_dtype : internal dtype for computation
    **kwargs: argument passthrough to the pairwise registration method.

    Returns
    -------
    Sequence registration model

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # These numbers are approximately spaced out as powers of the golden ratio, but are all prime numbers:
    # There is a beautiful reason for this, ask Loic.
    # magic_numbers = [1, 2, 3, 7, 11, 19, 29, 47, 79, 127, 199, 317, 521, 839, 1367]
    # factor = golden_ratio = 1.618033988749895
    magic_numbers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    factor = 2

    image_sequence = list(image_sequence)
    length = len(image_sequence)
    if max_scale is None:
        max_scale = len(magic_numbers) - 1
    max_scale = min(max_scale, int(log(length, factor)))
    min_scale = min(min_scale, max_scale)

    if full_reg_scale is None:
        full_reg_scale = max_scale

    with asection(f"Registering image sequence of length: {length}"):

        ndim = image_sequence[0].ndim

        uv_set = set()
        with asection(f"Enumerating pairwise registrations needed..."):

            for scale in range(min_scale, max_scale):
                for offset in range(0, magic_numbers[scale], max(1, magic_numbers[max(0, scale - 1)])):
                    for u in range(offset, length, magic_numbers[scale]):
                        v = u + magic_numbers[scale]
                        if v >= length:
                            continue
                        atuple = (u, v, scale < full_reg_scale)
                        if atuple not in uv_set:
                            # aprint(f"Pair: ({u}, {v}) added")
                            uv_set.add(atuple)
                        else:
                            aprint(f"Pair: ({u}, {v}) already considered, that's all good just info...")

        pairwise_models = []
        with asection(f"Computing pairwise registrations..."):

            # Current backend:
            current_backend = Backend.current()

            # function to process a subset of the pairwise registrations:
            def process(uv_list):
                with current_backend.copy():
                    for u, v, fullreg in uv_list:
                        image_u = image_sequence[u]
                        image_v = image_sequence[v]
                        _pairwise_registration(pairwise_models,
                                               u, v,
                                               image_u, image_v,
                                               min_confidence,
                                               mode,
                                               fullreg,
                                               internal_dtype,
                                               **kwargs)

            # Convenient function to split a sequence into approximately equal sized lists:
            def split_list(a_seq: Sequence, n):
                a_list = list(a_seq)
                k, m = divmod(len(a_list), n)
                return (a_list[_i * k + min(_i, m):(_i + 1) * k + min(_i + 1, m)] for _i in range(n))

            # Setup the number of workers:
            if workers == -1:
                workers = os.cpu_count() // 2
            aprint(f"Number of workers: {workers}")

            # We split the uv_set into approx. equal lists:
            uv_list_list = split_list(uv_set, workers)

            # Start jobs:
            if workers > 1:
                Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(uv_list) for uv_list in uv_list_list)
            else:
                for uv_list in uv_list_list:
                    process(uv_list)

        nb_models = len(pairwise_models)
        aprint(f"Number of models obtained: {nb_models} for a sequence of length:{length}")

        with asection(f"Solving for optimal sequence registration"):
            with NumpyBackend():
                xp = Backend.get_xp_module()
                sp = Backend.get_sp_module()

                if mode == 'translation':

                    # prepares list of models:
                    translation_models: List[TranslationRegistrationModel] = list(TranslationRegistrationModel(xp.zeros((ndim,), dtype=internal_dtype)) for _ in range(length))

                    for d in range(ndim):

                        # instantiates system to solve with zeros:
                        a = xp.zeros((nb_models + 1, length), dtype=xp.float32)
                        y = xp.zeros((nb_models + 1,), dtype=xp.float32)

                        # iterates through all pairwise registrations:
                        zero_vector = None
                        for i, model in enumerate(pairwise_models):
                            u = model.u
                            v = model.v

                            if i == 0:
                                # we make sure that all shifts are relative to the first timepoint:
                                zero_vector = model.shift_vector[d].copy()

                            vector = model.shift_vector[d] - zero_vector

                            # Each pairwise registration defines a
                            y[i] = vector.copy()
                            a[i, u] = +1
                            a[i, v] = -1

                        # Forces solution to have no displacement for first time point:
                        y[-1] = 0
                        a[-1, 0] = 1

                        # move matrix to sparse domain (much faster!):
                        a = sp.sparse.coo_matrix(a)

                        # solve system:
                        x_opt = linsolve(a, y,
                                         tolerance=1e-6,
                                         order_error=order_error,
                                         order_reg=order_reg,
                                         alpha_reg=alpha_reg)

                        # sets the shift vectors for the resulting sequence reg model:
                        for i in range(length):
                            translation_models[i].shift_vector[d] = -x_opt[i]

                    return SequenceRegistrationModel(model_list=translation_models)


                else:
                    raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")

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


def _pairwise_registration(pairwise_models,
                           u, v,
                           image_u, image_v,
                           min_confidence,
                           mode,
                           full_registration,
                           internal_dtype,
                           **kwargs):
    image_u = Backend.to_backend(image_u, dtype=internal_dtype)
    image_v = Backend.to_backend(image_v, dtype=internal_dtype)

    if mode == 'translation':
        model = register_translation_maxproj_nd(image_u, image_v,
                                                _display_phase_correlation=False,
                                                **kwargs)
        model.to_numpy()
        confidence = model.overall_confidence()

        if confidence >= min_confidence and full_registration:
            model.u = u
            model.v = v
            pairwise_models.append(model)
            # aprint(f"Success: registered images for indices: ({u}, {v}) with vector: {model.shift_vector} with confidence {confidence} (>={min_confidence}) ")
        else:
            aprint(f"Warning: low confidence ({confidence}) for pair: ({u}, {v})")
            com_u = center_of_mass(image_u, projection_type='max', offset_mode='middle')
            com_v = center_of_mass(image_v, projection_type='max', offset_mode='middle')
            model = TranslationRegistrationModel(shift_vector=com_u - com_v, confidence=1)
            model.to_numpy()
            model.u = u
            model.v = v
            pairwise_models.append(model)


    else:
        raise ValueError(f"Unsupported sequence stabilisation mode: {mode}")

    del image_u, image_v
